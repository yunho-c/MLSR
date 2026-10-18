// Stream a new Float32 WAV through the installed Go/C ABI package.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	mlsr "github.com/yunho-c/MLSR/bindings/go"
	"io"
	"math"
	"os"
)

func run() error {
	manifest := flag.String("manifest", "", "YAML manifest")
	bundle := flag.String("bundle", "", "model bundle")
	output := flag.String("output", "", "new WAV path")
	text := flag.String("text", "Hello from MLSR.", "text")
	language := flag.String("language", "en", "language")
	voice := flag.String("voice", "M1", "voice")
	cancelFirst := flag.Bool("cancel-first", false, "exercise cancellation before synthesis")
	flag.Parse()
	if *manifest == "" || *bundle == "" || *output == "" {
		return fmt.Errorf("manifest, bundle and output are required")
	}
	wav, err := os.OpenFile(*output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	complete := false
	defer func() {
		wav.Close()
		if !complete {
			os.Remove(*output)
		}
	}()
	if _, err = wav.Write(make([]byte, 56)); err != nil {
		return err
	}
	engine, err := mlsr.Open(*manifest, *bundle, mlsr.EngineOptions{})
	if err != nil {
		return err
	}
	defer engine.Close()
	req := mlsr.DefaultRequest(*text)
	req.Language = *language
	req.Voice = *voice
	req.Seed = 1729
	if *cancelFirst {
		first := req
		first.QueueFrames = 1
		first.ChunkFrames = 1
		ctx, cancel := context.WithCancel(context.Background())
		job, err := engine.Start(ctx, first)
		if err != nil {
			cancel()
			return err
		}
		_, err = job.Next()
		cancel()
		if err != nil {
			job.Close()
			return err
		}
		done, err := job.Wait(30000)
		if err != nil || !done {
			job.Close()
			return fmt.Errorf("cancellation did not quiesce: %v", err)
		}
		if err = job.Close(); err != nil {
			return err
		}
	}
	job, err := engine.Start(context.Background(), req)
	if err != nil {
		return err
	}
	defer job.Close()
	var frames uint64
	for {
		event, err := job.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if event.Kind == "audio" {
			frames += event.Frames
			if frames*4 > 0xffffffff-48 {
				return fmt.Errorf("classic WAV size limit exceeded")
			}
			bytes := make([]byte, len(event.Samples)*4)
			for i, value := range event.Samples {
				binary.LittleEndian.PutUint32(bytes[i*4:], math.Float32bits(value))
			}
			if _, err = wav.Write(bytes); err != nil {
				return err
			}
		}
		if event.Terminal() && event.Kind != "completed" {
			return fmt.Errorf("synthesis %s: %v", event.Kind, event.Failure)
		}
	}
	header := make([]byte, 56)
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], 48+uint32(frames*4))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 3)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], 44100)
	binary.LittleEndian.PutUint32(header[28:], 176400)
	binary.LittleEndian.PutUint16(header[32:], 4)
	binary.LittleEndian.PutUint16(header[34:], 32)
	copy(header[36:], "fact")
	binary.LittleEndian.PutUint32(header[40:], 4)
	binary.LittleEndian.PutUint32(header[44:], uint32(frames))
	copy(header[48:], "data")
	binary.LittleEndian.PutUint32(header[52:], uint32(frames*4))
	if _, err = wav.WriteAt(header, 0); err != nil {
		return err
	}
	if err = wav.Close(); err != nil {
		return err
	}
	complete = true
	fmt.Printf("Wrote %d frames to %s\n", frames, *output)
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
