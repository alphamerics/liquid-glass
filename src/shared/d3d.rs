//! Small shared D3D11 helpers: buffer creation, the standard sampler and
//! the process-wide shader compile cache.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::{
    Fxc::{D3DCompile, D3DCOMPILE_ENABLE_STRICTNESS},
    ID3DBlob,
};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Buffer, ID3D11Device, ID3D11SamplerState, D3D11_BIND_CONSTANT_BUFFER,
    D3D11_BUFFER_DESC, D3D11_COMPARISON_ALWAYS, D3D11_FILTER_MIN_MAG_MIP_LINEAR,
    D3D11_SAMPLER_DESC, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE_ADDRESS_CLAMP,
    D3D11_USAGE_DEFAULT,
};

pub(crate) fn constant_buffer<T>(device: &ID3D11Device, value: &T) -> Result<ID3D11Buffer, String> {
    let mut buffer = None;
    unsafe {
        device.CreateBuffer(
            &D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<T>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
                StructureByteStride: 0,
            },
            Some(&D3D11_SUBRESOURCE_DATA {
                pSysMem: (value as *const T).cast(),
                SysMemPitch: 0,
                SysMemSlicePitch: 0,
            }),
            Some(&mut buffer),
        )
    }
    .map_err(|error| format!("Could not create a constant buffer: {error}"))?;
    Ok(buffer.expect("D3D11 constant buffer missing"))
}

pub(crate) fn create_linear_clamp_sampler(device: &ID3D11Device) -> ID3D11SamplerState {
    let mut sampler = None;
    unsafe {
        device
            .CreateSamplerState(
                &D3D11_SAMPLER_DESC {
                    Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                    AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
                    AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
                    MipLODBias: 0.0,
                    MaxAnisotropy: 1,
                    ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                    BorderColor: [0.0; 4],
                    MinLOD: 0.0,
                    MaxLOD: f32::MAX,
                },
                Some(&mut sampler),
            )
            .expect("Could not create the D3D11 sampler");
    }
    sampler.expect("D3D11 sampler missing")
}

/// Compiles a shader program once per (source, entry) pair for the whole
/// process: spawning a surface must never re-run the HLSL compiler.
pub(crate) fn compiled_shader(
    source: &'static str,
    entry: &str,
    target: &str,
) -> Result<Vec<u8>, String> {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, String), Vec<u8>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("shader cache lock poisoned");
    if let Some(blob) = cache.get(&(source, entry.to_owned())) {
        return Ok(blob.clone());
    }
    let blob = compile_shader_from(source, entry, target)?;
    cache.insert((source, entry.to_owned()), blob.clone());
    Ok(blob)
}

/// Compiles HLSL source to shader bytecode via D3DCompile.
pub(crate) fn compile_shader_from(
    source: &str,
    entry: &str,
    target: &str,
) -> Result<Vec<u8>, String> {
    let entry = CString::new(entry).expect("shader entry contains no NUL");
    let target = CString::new(target).expect("shader target contains no NUL");
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None::<&windows::Win32::Graphics::Direct3D::ID3DInclude>,
            PCSTR(entry.as_ptr() as _),
            PCSTR(target.as_ptr() as _),
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &mut blob,
            Some(&mut errors),
        )
        .map_err(|error| {
            let details = errors
                .as_ref()
                .map(|blob| {
                    let bytes = std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    );
                    String::from_utf8_lossy(bytes).into_owned()
                })
                .unwrap_or_default();
            format!("Could not compile the D3D11 liquid shader: {error}\n{details}")
        })?;
        let blob = blob.expect("D3D compiler returned no bytecode");
        Ok(
            std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())
                .to_vec(),
        )
    }
}
