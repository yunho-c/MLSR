// Export with the shared bounded writer; optional native playback stays in Rust.
package main

import (
	"context"
	"flag"
	"fmt"
	"github.com/yunho-c/MLSR/bindings/go"
	"os"
	"os/signal"
)

func run(ctx context.Context) error {
	manifest := flag.String("manifest", "", "pinned YAML manifest")
	bundle := flag.String("bundle", "", "verified local model directory")
	output := flag.String("output", "", "new WAV path (never overwritten)")
	speak := flag.Bool("speak", false, "play through native output before export")
	policy := flag.String("policy", "cpu_and_neural_engine", "cpu_only, cpu_and_neural_engine or qnn_htp")
	flag.Parse()
	if *manifest == "" || *bundle == "" || *output == "" {
		return fmt.Errorf("--manifest, --bundle and --output are required")
	}
	options := mlsr.DefaultTTSOptions()
	if *policy != "cpu_only" && *policy != "cpu_and_neural_engine" && *policy != "qnn_htp" {
		return fmt.Errorf("unknown compute policy %q", *policy)
	}
	if !*speak {
		options.Output = mlsr.HeadlessOutput
	}
	tts, err := mlsr.OpenTTS(ctx, *manifest, *bundle, mlsr.EngineOptions{CPUOnly: *policy == "cpu_only", QNNHTP: *policy == "qnn_htp"}, options)
	if err != nil {
		return err
	}
	defer tts.Close()
	request := mlsr.DefaultSpeechRequest("Hello from MLSR.")
	request.Backend.Seed = 1729
	var file mlsr.WavFile
	if *speak {
		speech, err := tts.Speak(ctx, request, mlsr.DefaultSpeechOptions(), mlsr.Replace)
		if err != nil {
			return err
		}
		defer speech.Close()
		if _, err := speech.Wait(ctx); err != nil {
			return err
		}
		file, err = tts.WriteSpeech(speech, *output, 60*44100)
		if err != nil {
			return err
		}
	} else {
		// Context cancellation reaches the independently owned job while its
		// writer consumes bounded chunks, then joins native cleanup.
		file, err = tts.RenderToFile(ctx, request, *output, 60*44100)
		if err != nil {
			return err
		}
	}
	fmt.Printf("Wrote %d frames to %s\n", file.Frames, file.Path)
	return nil
}
func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()
	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
