# VRM avatars for testing (E3)

Two sample avatars from pixiv's three-vrm repository, both licensed for
redistribution and modification by their own VRM metadata (quoted below as
stored in each file). Any other `.vrm` works too: `--vrm PATH`.

## VRM1_Constraint_Twist_Sample.vrm (VRM 1.0, "Seed-san")

- Source: https://github.com/pixiv/three-vrm, branch `dev`, commit
  3caf7a21ee68f0d5dfa363ab3d577450bbf60136,
  `packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm`
- sha256 `12c2b97e95e700783a6a550dc0eee2d7880aeedccef9ae67bc4c5a2f0f2631a2`,
  10 776 032 bytes
- `VRMC_vrm.meta`: name "VRM1_Constraint_Twist_Sample", version v1.0.1,
  authors ["pixiv Inc."], copyrightInformation "(c) 2022 pixiv Inc.",
  licenseUrl https://vrm.dev/licenses/1.0/ (VRM Public License 1.0),
  avatarPermission "everyone", commercialUsage "corporation",
  creditNotation "unnecessary", allowRedistribution true, modification
  "allowModificationRedistribution", allowExcessivelyViolentUsage true,
  allowExcessivelySexualUsage true, allowPoliticalOrReligiousUsage true,
  allowAntisocialOrHateUsage false.
- Body file: `rigs/seed-san.body.json` (`python -m flow.vrmbody --vrm
  assets/vrm/VRM1_Constraint_Twist_Sample.vrm --out rigs/seed-san.body.json`).

## three-vrm-girl.vrm (VRM 0.x)

- Source: https://github.com/pixiv/three-vrm, branch `0.x`,
  `packages/three-vrm/examples/models/three-vrm-girl.vrm` (blob
  5a3a146d5ac90de6c37986f8fe6b88b91322ae8d)
- sha256 `e6e179dea8bafc712900b5012628166a64e982d45566fb705d2ec52f579279b9`,
  5 558 600 bytes
- `VRM.meta`: title "three-vrm-girl", version 1.1, author "pixiv Inc.",
  licenseName "Other", otherLicenseUrl / otherPermissionUrl
  https://hub.vroid.com/license?allowed_to_use_user=everyone&characterization_allowed_user=everyone&corporate_commercial_use=allow&credit=unnecessary&modification=allow&personal_commercial_use=profit&redistribution=allow&sexual_expression=allow&version=1&violent_expression=allow
  (VRoid Hub conditions: everyone, redistribution allowed, modification
  allowed, credit unnecessary, commercial use allowed).
- Kept to test the VRM 0 path (the avatar faces -Z; tests/test_vrm.py).
