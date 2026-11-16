//! The capture frame's plumbing: the stable shared texture creation and
//! the capture scale math.

use windows::Graphics::SizeInt32;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11ShaderResourceView, ID3D11Texture2D, D3D11_BIND_SHADER_RESOURCE,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};

use crate::shared::CAPTURE_SCALE;
pub(super) fn scale_size(size: SizeInt32) -> SizeInt32 {
    SizeInt32 {
        Width: ((size.Width as f32 * CAPTURE_SCALE).round() as i32).max(1),
        Height: ((size.Height as f32 * CAPTURE_SCALE).round() as i32).max(1),
    }
}

/// The shared capture frame texture plus its shader view, created together:
/// every consumer samples the same view for the texture's whole lifetime
/// (recreated only when the monitor's size changes). Returns a `Result`
/// rather than panicking: this runs on the WGC callback thread, and the
/// release profile aborts the whole process on any panic — a D3D failure
/// (a driver reset, most commonly) must degrade to a skipped frame
/// instead.
pub(super) fn create_frame_texture(
    device: &ID3D11Device,
    size: SizeInt32,
) -> Result<(ID3D11Texture2D, ID3D11ShaderResourceView), String> {
    let mut texture = None;
    unsafe {
        device.CreateTexture2D(
            &D3D11_TEXTURE2D_DESC {
                Width: size.Width as u32,
                Height: size.Height as u32,
                MipLevels: 1,
                ArraySize: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            },
            None,
            Some(&mut texture),
        )
    }
    .map_err(|error| format!("Could not create the shared capture frame texture: {error}"))?;
    let texture = texture.expect("D3D11 shared capture frame texture missing");
    let mut view = None;
    unsafe {
        device
            .CreateShaderResourceView(&texture, None, Some(&mut view))
            .map_err(|error| format!("Could not create the shared capture frame view: {error}"))?;
    }
    Ok((
        texture,
        view.expect("D3D11 shared capture frame view missing"),
    ))
}
