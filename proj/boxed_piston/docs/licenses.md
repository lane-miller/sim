# Dependency licenses (engineering screen)

**Not legal advice; engineering screen only.** This document records third-party
packages boxrad may depend on, how they are used, and the license posture we
target for new dependencies.

**Policy:** accept permissive licenses (MIT, BSD, Apache-2.0) or unmodified,
dynamically loaded LGPL only (e.g. Open CASCADE via cadquery-ocp-novtk). Prefer
in-house code when it can meet roughly 85% of the need. Record every new
dependency in this file after verification.

Verified: **2026-11-21**.

## In use / planned

| Package | Role | Extra | Version verified | License | Source URL | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| numpy | Core numerics | (base) | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://pypi.org/project/numpy/ | |
| scipy | Core numerics | (base) | 1.17.1 | BSD-3-Clause | https://pypi.org/project/scipy/ | |
| numba | JIT kernels | (base) | 0.65.1 | BSD-2-Clause | https://pypi.org/project/numba/ | Pulls llvmlite |
| llvmlite | LLVM bindings (via numba) | (base, transitive) | 0.47.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | https://pypi.org/project/llvmlite/ | |
| h5py | HDF5 I/O | (base) | 3.16.0 | BSD-3-Clause | https://pypi.org/project/h5py/ | |
| matplotlib | Plotting / viz CLI | (base) | 3.10.9 | Matplotlib License (PSF-based, permissive) | https://pypi.org/project/matplotlib/ | |
| manifold3d | Mesh booleans, manifold check, plane slices, polygon triangulation | `geom` | 3.5.5 | Apache-2.0 | https://github.com/elalish/manifold | No required runtime deps |
| cadquery-ocp-novtk | STEP import only (Open CASCADE Python bindings) | `cad` | 8.0.1.1.0 | Apache-2.0 (bindings); OCCT LGPL-2.1-only WITH OCCT-exception-1.0 (bundled) | https://pypi.org/project/cadquery-ocp-novtk/ | Bundles OCCT 8.0.1 shared libraries (`libTK*.8.0.1`); wheel ships no OCCT license file. OCCT license CONFIRMED 2026-11-21: LGPL-2.1 with Open CASCADE Exception 1.0 (SPDX: LGPL-2.1-only WITH OCCT-exception-1.0), per OCCT repo tag V8_0_1 (`LICENSE_LGPL_21.txt`, `OCCT_LGPL_EXCEPTION.txt`). Exception requires prominent notice of OCCT use in supporting docs: satisfied by `NOTICE`. |
| bempp-cl | BEM validation reference | `bem` | 0.4.2 | MIT | https://pypi.org/project/bempp-cl/ | Validation only; never use `bempp_cl.api.shapes` (gmsh) |

### Libraries bundled in cadquery-ocp-novtk (unmodified, dynamically loaded; not used directly by boxrad)

- **Permissive:** zlib, zstd, liblzma, libpng, libjpeg, libtiff, libwebp/webpmux/sharpyuv, OpenEXR/Imath/Iex/IlmThread, Lerc (Apache-2.0), openjp2, openjph, jpegxr, lcms2, libdeflate, fmt, libc++ (Apache-2.0 WITH LLVM-exception).
- **Dual-licensed, used under the non-GPL option:** FreeImage (FreeImage Public License), FreeType (FreeType License), LibRaw (LGPL-2.1 / CDDL-1.0).
- **cadquery-ocp-proxy** 8.0.1.1.0: metadata-only proxy package pulled by the OCP wheel; contains no code.

Run `python tools/license_report.py` in an environment with the relevant extras
installed to print installed versions and bundled license file excerpts.

## Considered, not used

| Package | License | Why not |
| --- | --- | --- |
| shapely | BSD-3-Clause; bundles **GEOS** (LGPL-2.1) | Replaced by in-house geometry (`boxrad/geom`, planned) to minimize license surface |
| trimesh | MIT | Same; in-house mesh/IO instead |

vtk: not used; cadquery-ocp (VTK build) avoided because it pins vtk==9.6.2.

## Forbidden (do not add)

- **GPL family:** gmsh, pygmsh, pymeshlab (GPL-3.0), CGAL-based tools including pygalmesh, and similar copyleft mesh/CAD stacks unless explicitly re-approved.
- **Non-commercial / research-only:** Shewchuk Triangle (`triangle`, `meshpy`), TetGen, and anything with non-commercial or paid terms.
- **Unused full CAD stack:** `cadquery` (we use `cadquery-ocp-novtk` only for STEP).
- **Paid or non-commercial** packages of any kind.

---

*Not legal advice; engineering screen.*