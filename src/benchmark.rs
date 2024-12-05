use std::path::Path;

use criterion::{black_box, Criterion};
use miette::{miette, Context, IntoDiagnostic, Result};
use openvino::CompiledModel;

fn load_model(path: impl AsRef<Path>) -> Result<CompiledModel> {
    let mut core = openvino::Core::new().into_diagnostic()?;
    let onnx_path = path
        .as_ref()
        .to_str()
        .ok_or_else(|| miette!("Model file path is not valid UTF-8"))?;
    let model = core.read_model_from_file(onnx_path, "").into_diagnostic()?;
    core.compile_model(&model, openvino::DeviceType::CPU)
        .into_diagnostic()
}

/// Run a benchmark on an ONNX model.
pub fn run_benchmark_onnx(path: impl AsRef<Path>, c: &mut Criterion) -> Result<()> {
    let mut model = load_model(&path).context("failed to load model")?;
    let onnx_name = path
        .as_ref()
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| miette!("model file name is not valid UTF-8"))?;

    c.bench_function(onnx_name, |b| {
        b.iter(|| {
            let request = model
                .create_infer_request()
                .expect("failed to create inference request");

            black_box(request)
                .infer()
                .expect("failed to complete inference");
        });
    });

    Ok(())
}
