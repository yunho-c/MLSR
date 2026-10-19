package mlsr

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSynthesisOwnedError(t *testing.T) {
	before := nativeHandleCount()
	_, err := Open("/missing/mlsr-manifest.yaml", "/missing/mlsr-bundle", EngineOptions{})
	var native *Error
	if !errors.As(err, &native) || native.Code != "io_error" {
		t.Fatalf("wrong error: %v", err)
	}
	if nativeHandleCount() != before {
		t.Fatal("error handle leaked")
	}
}
func TestRealSynthesis(t *testing.T) {
	if os.Getenv("MLSR_TEST_BUNDLE") == "" {
		t.Skip("real model assets not requested")
	}
	before := nativeHandleCount()
	engine, err := Open(os.Getenv("MLSR_TEST_MANIFEST"), os.Getenv("MLSR_TEST_BUNDLE"), EngineOptions{QNNHTP: os.Getenv("MLSR_TEST_POLICY") == "qnn_htp"})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := DefaultRequest("Hello from MLSR. Another sentence.")
	req.QueueFrames = 1
	req.ChunkFrames = 1
	job, err := engine.Start(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if event, err := job.Next(); err != nil || event.Kind != "predicted" {
		t.Fatalf("prediction: %v %v", event, err)
	}
	if busy, err := engine.Start(context.Background(), req); err == nil {
		busy.Close()
		t.Fatal("accepted busy request")
	}
	cancel() // Context watcher must cancel even when Next is not called.
	if done, err := job.Wait(30000); err != nil || !done {
		t.Fatalf("cancel wait: %v %v", done, err)
	}
	if event, err := job.Next(); err != nil || event.Kind != "cancelled" {
		t.Fatalf("cancel outcome: %v %v", event, err)
	}
	if err := job.Close(); err != nil {
		t.Fatal(err)
	}
	output := os.Getenv("MLSR_TEST_OUTPUT")
	if output != "" {
		if err := os.MkdirAll(filepath.Dir(output), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(output, 0755); err != nil {
			t.Fatal(err)
		}
	}
	var retained []float32
	for index, pair := range [][2]string{{"en", "M1"}, {"en", "F1"}, {"ko", "M1"}, {"ko", "F1"}} {
		text := "Hello from MLSR."
		if pair[0] == "ko" {
			text = "안녕하세요. 반갑습니다."
		}
		req := DefaultRequest(text)
		req.Language = pair[0]
		req.Voice = pair[1]
		req.Seed = uint64(1729 + index)
		job, err := engine.Start(context.Background(), req)
		if err != nil {
			t.Fatal(err)
		}
		var frames uint64
		var pcm []byte
		terminals := 0
		for {
			event, err := job.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				job.Close()
				t.Fatal(err)
			}
			if event.JobID != job.ID {
				t.Fatal("wrong job ID")
			}
			if event.Kind == "audio" {
				if event.StartFrame != frames || event.Frames > 4096 || event.SampleRate != 44100 || event.Channels != 1 {
					t.Fatal("bad PCM metadata")
				}
				if event.TextEnd > uint64(len(text)) || event.TextStart >= event.TextEnd {
					t.Fatal("bad source span")
				}
				frames += event.Frames
				retained = event.Samples
				for _, sample := range event.Samples {
					pcm = binary.LittleEndian.AppendUint32(pcm, math.Float32bits(sample))
				}
			}
			if event.Terminal() {
				terminals++
				if event.Kind != "completed" || event.Frames != frames {
					t.Fatalf("bad terminal: %v", event)
				}
			}
		}
		if err := job.Close(); err != nil {
			t.Fatal(err)
		}
		if frames < 44100 || terminals != 1 {
			t.Fatal("incomplete speech")
		}
		if output != "" {
			if err := os.WriteFile(filepath.Join(output, pair[0]+"_"+pair[1]+".f32"), pcm, 0644); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := engine.Close(); err != nil {
		t.Fatal(err)
	}
	if len(retained) == 0 || nativeHandleCount() != before {
		t.Fatal("output lifetime or handle cleanup failed")
	}
}
