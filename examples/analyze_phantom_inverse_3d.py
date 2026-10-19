"""Vessel-core metrics, comparison plots, and a standalone report body."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def analyze(folder: Path) -> None:
    results = json.loads((folder / "results.json").read_text())
    rows, volumes = [], []
    for stage in results["stages"]:
        data = np.load(folder / f"stage-{stage['waves']:g}.npz")
        truth, fitted, window = (
            data["truth"],
            data["reconstruction"],
            data["window"],
        )
        interior = window > 0.99
        normalized = data["points"] + [0.45, 0, 0.15 - 0.6875]
        interior &= (
            np.linalg.norm(normalized / [0.92, 0.82, 1.0], axis=-1) < 0.84
        )
        target = (truth > 0.04) & interior
        predicted = (fitted > 0.04) & interior
        tp = int(np.sum(target & predicted))
        fp = int(np.sum(~target & predicted))
        fn = int(np.sum(target & ~predicted))
        rows.append(
            dict(
                **stage,
                core_dice=2 * tp / max(1, 2 * tp + fp + fn),
                core_precision=tp / max(1, tp + fp),
                core_recall=tp / max(1, tp + fn),
                true_core_voxels=int(target.sum()),
                predicted_core_voxels=int(predicted.sum()),
            )
        )
        volumes.append(np.round(fitted * 10000).astype(int).ravel().tolist())
    (folder / "analysis.json").write_text(json.dumps(rows, indent=2))
    n = truth.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), layout="constrained")
    for row, projection_axis in enumerate((1, 0)):
        for col, (array, label) in enumerate(
            ((truth, "True phantom"), (fitted, "Reconstructed"))
        ):
            speed = 1500 / np.sqrt(1 - array)
            shown = np.max(speed, axis=projection_axis).T
            im = axes[row, col].imshow(
                shown,
                origin="lower",
                extent=[-8, 8, -8, 8],
                vmin=1460,
                vmax=1590,
                cmap="viridis",
            )
            axes[row, col].set_title(f"{label}: view along {'yx'[row]}")
        error = np.max(
            np.abs(1500 / np.sqrt(1 - fitted) - 1500 / np.sqrt(1 - truth)),
            axis=projection_axis,
        ).T
        err = axes[row, 2].imshow(
            error,
            origin="lower",
            extent=[-8, 8, -8, 8],
            vmin=0,
            vmax=100,
            cmap="magma",
        )
        axes[row, 2].set_title("Maximum absolute error")
        for ax in axes[row]:
            ax.set_xlabel(f"{'xy'[row]} (mm from crop center)")
            ax.set_ylabel("z (mm from crop center)")
    fig.colorbar(im, ax=axes[:, :2], label="Sound speed (m/s)", shrink=0.7)
    fig.colorbar(err, ax=axes[:, 2], label="Error (m/s)", shrink=0.7)
    fig.suptitle(
        f"Measurement-only full-wave fit | {rows[-1]['wavelength_mm']:.2f} mm wavelength"
    )
    fig.savefig(folder / "reconstruction.png", dpi=180)
    fig.savefig(folder / "reconstruction.pdf")
    plt.close(fig)

    options = "".join(
        f'<option value="{i}" {"selected" if i == len(rows) - 1 else ""}>{r["wavelength_mm"]:.2f} mm wavelength</option>'
        for i, r in enumerate(rows)
    )
    table = "".join(
        f'<tr><th scope="row">{r["wavelength_mm"]:.2f} mm</th><td>{r["relative_data_error"]:.2%}</td><td>{r["relative_image_error"]:.2%}</td><td>{r["core_dice"]:.3f}</td><td>{r["core_precision"]:.1%}</td><td>{r["core_recall"]:.1%}</td><td>{r["seconds"]:.0f} s</td></tr>'
        for r in rows
    )
    payload = dict(
        n=n,
        truth=np.round(truth * 10000).astype(int).ravel().tolist(),
        fits=volumes,
        colors=np.round(
            255 * plt.get_cmap("viridis")(np.linspace(0, 1, 256))[:, :3]
        )
        .astype(int)
        .tolist(),
    )
    body = """
<section class="a-section"><h2 class="a-section__title">What was tested</h2>
<p class="a-prose">A 16 × 16 × 16 mm crop from the same synthetic branching breast phantom was reconstructed from noiseless exterior wave measurements. The optimizer started from water (zero contrast) and fit a free value at every voxel. It never saw the target image or vessel labels. This is full-wave optimization, not neural-network training.</p>
<p class="a-prose">The crop was smoothly tapered at its boundary and isolated from the rest of the breast. Transmitters and receivers surrounded it. These deliberately favorable conditions test whether local recovery is possible; they do not establish recovery in the complete breast or with measurement noise.</p></section>
<section class="a-section"><h2 class="a-section__title">Inspect the reconstruction</h2>
<div style="display:flex;flex-wrap:wrap;gap:1rem;margin-bottom:1rem">
<label>Stage <select id="stage">OPTIONS</select></label>
<label>View along <select id="axis"><option value="0">x</option><option value="1" selected>y</option><option value="2">z</option></select></label>
<label>Display <select id="mode"><option value="projection">Maximum-intensity projection</option><option value="slice">Single slice</option></select></label>
<label>Slice <input id="slice" type="range" min="0" max="NMAX" value="NMID" disabled></label></div>
<div class="a-grid a-grid--halves"><figure><figcaption>True phantom</figcaption><canvas id="truth" width="N" height="N" aria-label="True sound-speed volume" style="width:100%;image-rendering:pixelated"></canvas></figure><figure><figcaption>Fitted from wave measurements</figcaption><canvas id="fit" width="N" height="N" aria-label="Reconstructed sound-speed volume" style="width:100%;image-rendering:pixelated"></canvas></figure></div>
<canvas id="colorbar" width="256" height="14" aria-label="Common sound-speed scale" style="width:100%;height:14px"></canvas><div style="display:flex;justify-content:space-between"><span>1460 m/s</span><span>Common color scale</span><span>1590 m/s</span></div><p id="viewlabel" class="a-section__note" aria-live="polite"></p></section>
<section class="a-section"><h2 class="a-section__title">Accuracy across wavelengths</h2><div class="a-table-scroll"><table class="a-table"><caption>Each stage continues the same reconstruction at a shorter wavelength</caption><thead><tr><th scope="col">Wavelength</th><th scope="col">Measurement error</th><th scope="col">Contrast image error</th><th scope="col">Vessel-core Dice</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Stage time</th></tr></thead><tbody>TABLE</tbody></table></div>
<p class="a-section__note">Errors are relative L2 norms. Vessel-core masks use the same fixed b &gt; 0.04 threshold for truth and reconstruction, restricted to the untapered interior (window &gt; 0.99) and normalized breast radius below 0.84, excluding skin. This threshold corresponds to sound speed above 1530.9 m/s; glandular tissue alone is below it. Dice is voxel overlap, not a count of separately recovered branches. The complete geometric vessel supports are wider than these high-contrast cores.</p></section>
<section class="a-section"><h2 class="a-section__title">Experiment details and limits</h2><p class="a-prose">PARAMETERS</p>
<p class="a-prose">L-BFGS-B minimized normalized complex pressure mismatch plus a quadratic penalty on neighboring voxel differences. Contrast b = 1 − (1500/c)² was bounded between −0.1 and 0.2 before multiplication by the known taper. The forward model solves the full 3D Lippmann–Schwinger equation with FFT convolution and GMRES; derivatives use implicit differentiation. The forward implementation agrees with the existing solver and its gradient passed a finite-difference check.</p>
<p class="a-prose">The fitting data and optimizer used the same grid and forward model. Low training residual alone can therefore overstate recovery. Independent directions and finer-grid checks are reported separately below. No robustness, clinical resolution, neural-network generalization, or whole-breast recovery claim follows from this experiment.</p><div id="validation-note">VALIDATION</div></section>
<script type="application/json" id="volume-data">DATA</script>
<script>
(()=>{const d=JSON.parse(document.getElementById('volume-data').textContent), n=d.n;
const controls=['stage','axis','mode','slice'].map(id=>document.getElementById(id));
function draw(){const axis=+controls[1].value, slice=+controls[3].value, projection=controls[2].value==='projection';controls[3].disabled=projection;
for(const [id,values] of [['truth',d.truth],['fit',d.fits[+controls[0].value]]]){const ctx=document.getElementById(id).getContext('2d'), img=ctx.createImageData(n,n);
for(let row=0;row<n;row++)for(let col=0;col<n;col++){let b=-1;const p=[0,0,0], other=[0,1,2].filter(a=>a!==axis);p[other[0]]=col;p[other[1]]=n-1-row;
for(let s=projection?0:slice;s<(projection?n:slice+1);s++){p[axis]=s;b=Math.max(b,values[p[0]*n*n+p[1]*n+p[2]]/10000);}
const c=1500/Math.sqrt(1-b), k=Math.max(0,Math.min(255,Math.round((c-1460)/130*255))), at=4*(row*n+col);img.data.set([...d.colors[k],255],at);}ctx.putImageData(img,0,0);}
document.getElementById('viewlabel').textContent=(projection?'Maximum-intensity projection along '+ 'xyz'[axis]:('xyz'[axis]+' = '+(-8+16*(slice+.5)/n).toFixed(2)+' mm from crop center'))+'. Each image spans 16 mm × 16 mm.';}
controls.forEach(e=>e.addEventListener('input',draw));const ctx=document.getElementById('colorbar').getContext('2d');d.colors.forEach((c,i)=>{ctx.fillStyle='rgb('+c.join(',')+')';ctx.fillRect(i,0,1,14);});draw();})();
</script>
"""
    validation_path = folder / "validation.json"
    validation = "Finer-grid and unused-direction checks have not been run."
    if validation_path.exists():
        checked = json.loads(validation_path.read_text())
        validation = '<h3 class="a-section__title">Unused directions and grid refinement</h3><ul>'
        for row in checked["grids"]:
            change = (
                f"; true-data change from previous grid {row['truth_change_from_previous_grid']:.2%}"
                if "truth_change_from_previous_grid" in row
                else ""
            )
            validation += f"<li>{row['n']}³ grid: measurement error {row['relative_holdout_error']:.2%}{change}.</li>"
        validation += '</ul><p class="a-section__note">Nine unused transmitter directions and 128 new receiver positions. The fitted image is linearly interpolated for evaluation on finer grids; it is not retrained.</p>'
    params = results["parameters"]
    description = f"{n}³ voxels ({16 / n:.3f} mm spacing), {params['sources']} plane-wave transmitters, and {params['receivers']} receivers on a surrounding sphere. Smoothness weight {params['regularization']:g}; up to {params['iterations']} iterations per wavelength. One NVIDIA A100 GPU. Maximum forward residual across fitting stages: {max(r['max_forward_residual'] for r in rows):.2g}."
    for key, value in {
        "OPTIONS": options,
        "NMAX": str(n - 1),
        "NMID": str(n // 2),
        'width="N"': f'width="{n}"',
        'height="N"': f'height="{n}"',
        "TABLE": table,
        "PARAMETERS": description,
        "VALIDATION": validation,
        "DATA": json.dumps(payload, separators=(",", ":")),
    }.items():
        body = body.replace(key, value)
    (folder / "report-body.html").write_text(body)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    analyze(parser.parse_args().folder)
