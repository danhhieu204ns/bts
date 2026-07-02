# Data Report

Scenes found: **13**

| Scene | Root | Train images | Test poses | Test GT | Sparse images | Train registered | Test registered | Image size | CSV sizes | Camera |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| hcm0031 | `phase1/public_set` | 200 | 50 | 50 | 388 | 200 | 50 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| hcm0034 | `phase1/public_set` | 240 | 60 | 60 | 337 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0181 | `phase1/public_set` | 240 | 60 | 60 | 371 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0193 | `phase1/public_set` | 240 | 60 | 60 | 397 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0204 | `phase1/public_set` | 240 | 60 | 60 | 409 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0249 | `phase1/private_set1` | 240 | 60 | 0 | 306 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0254 | `phase1/private_set1` | 240 | 60 | 0 | 341 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM0276 | `phase1/private_set1` | 240 | 60 | 0 | 372 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HCM1439 | `phase1/private_set1` | 103 | 26 | 0 | 129 | 103 | 26 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HNI0131 | `phase1/private_set1` | 240 | 60 | 0 | 466 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HNI0265 | `phase1/private_set1` | 205 | 52 | 0 | 257 | 205 | 52 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HNI0366 | `phase1/private_set1` | 240 | 60 | 0 | 357 | 240 | 60 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |
| HNI0437 | `phase1/private_set1` | 224 | 56 | 0 | 280 | 224 | 56 | (1320, 989) | [(1320, 989)] | SIMPLE_RADIAL 1320x989 |

## Pose Notes

`test_poses.csv` values `tx,ty,tz` align with COLMAP world-to-camera translation `tvec`. To compare camera locations, convert to center `C = -R^T t`.
- `hcm0031` sparse camera center ranges: (-7.715..7.567, -9.986..4.343, -9.391..5.246)
- `hcm0034` sparse camera center ranges: (-7.503..7.653, -9.262..3.740, -8.290..4.061)
- `HCM0181` sparse camera center ranges: (-7.133..7.430, -9.194..4.055, -8.167..5.077)
