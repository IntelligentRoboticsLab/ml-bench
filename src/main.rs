use std::path::PathBuf;

use clap::{command, Parser};
use criterion::PlottingBackend;

mod benchmark;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the onnx model file.
    model_path: PathBuf,
    /// The number of samples to collect for each benchmark.
    #[arg(short, long, default_value = "5000")]
    samples: usize,
}

fn main() -> miette::Result<()> {
    miette::set_panic_hook();
    let args = Args::parse();

    let model_path = args.model_path;
    if !model_path.exists() {
        eprintln!("Model file not found: {model_path:?}");
        std::process::exit(1);
    }

    let Some(model_path_str) = model_path.to_str() else {
        eprintln!("Model file path is not valid UTF-8: {model_path:?}");
        std::process::exit(1);
    };

    let mut criterion: ::criterion::Criterion<_> = ::criterion::Criterion::default()
        .sample_size(args.samples)
        // this is set to `None` to avoid having to pass the `--bench` argument
        .profile_time(None)
        .plotting_backend(PlottingBackend::Plotters);

    benchmark::run_benchmark_onnx(model_path_str, &mut criterion)
}
