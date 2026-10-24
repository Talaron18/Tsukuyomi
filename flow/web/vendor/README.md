# Vendored browser libraries

Fetched from the npm registry (registry.npmjs.org tarballs) and copied with
their package layout kept, so the relative imports inside them resolve
(`three.module.min.js` imports `./three.core.min.js`; `GLTFLoader.js`
imports `../utils/BufferGeometryUtils.js`). The pages map the bare names
with an import map: `three` -> `three/build/three.module.min.js`,
`three/addons/` -> `three/examples/jsm/`, `@pixiv/three-vrm` ->
`three-vrm/three-vrm.module.min.js`.

| package | version | published | license |
|---|---|---|---|
| three | 0.180.0 | 2025-09-03 | MIT (three/LICENSE) |
| @pixiv/three-vrm | 3.5.5 | 2026-07-09 | MIT (three-vrm/LICENSE) |

three 0.180 is the version three-vrm 3.5.5 is built and tested against (its
devDependency `three ^0.180.0`; peer `>=0.137`).

sha256:

    99f943e956e52dbf52bbe418899e8a8ae5e35f5c79a94ab8e404926afc620bf2  three-vrm/three-vrm.module.min.js
    279ec82987aec7e72ecb9850bb704a87e352d577bc7581454c30f3551a88ea92  three-vrm/LICENSE
    e2b5ee6bccd38fd6d8a2428546b83c5f2426d84b152ef82be8055556e3b40eb6  three/build/three.module.min.js
    61ba0df005b05991361d040d8ff670e1aadfd0ce7aeebd1fdb0725957a8957de  three/build/three.core.min.js
    67ac5551fdafa6e349bd80c8f8e5e39c136d6b2fb1ad647db9abb21dc86f9e4a  three/examples/jsm/loaders/GLTFLoader.js
    fda7e946b8e0b5ab39b779206589e7a1079a22eb24efb89d7223e03fdfb1f751  three/examples/jsm/utils/BufferGeometryUtils.js
    bfe119ea4fd413f5f7ca3fcd63adb0c4a073ed39daa2fe7d3e6b769e21272601  three/LICENSE
