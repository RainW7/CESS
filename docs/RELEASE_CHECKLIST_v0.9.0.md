# CESS v0.9.0 Release Checklist

[English](RELEASE_CHECKLIST_v0.9.0.md) | [简体中文](RELEASE_CHECKLIST_v0.9.0_zh-CN.md)

2026-10-26. The current release work concerns one-dimensional user acceptance and publication. CESS+ and 2D kinematics belong to a separate workstream.
The public outlook is in the [CESS+ roadmap](ROADMAP.md); detailed historical handover records remain local.

## Release decisions already established

- Project source and documentation use [MIT](../LICENSE), with CESS contributors as the copyright holder.
- The maintainer has approved distribution of the current throughput, background, and morphology resources; see the [resource notice](../RESOURCE_NOTICE.md) and [inventory](RESOURCE_MANIFEST.json).
- “Latest” throughput reflects the maintainer's description of the supplied copies. Record their SHA values without inventing an official release date or data license.
- The production path does not call the old `varsmooth`. Its former upstream implementation has been removed; the compatibility method uses independently implemented Gaussian integration. Historical research snapshots are not distributed in the source package.
- Real BayeSED catalogues, scientific outputs, private paths, environment directories, and development checkpoints are excluded.

The release source version is now 0.9.0. The user completed the selected whole-file and six-input directory calculations, followed by checks of the concurrency improvements. Final validation of the exact 0.9.0 source and artifacts is recorded separately; older development-version evidence is not substituted for it. Manifests cannot be resumed across code identities.

## User acceptance

Run the selected HDF5 in full and check source counts, IDs/row mappings, normal-source spectra, invalid-row flags, and product readability.
After user acceptance, the maintainer and release collaborator finalize the version. If a new issue appears, fix it and update the relevant evidence first.
The user has reported completion of whole-file and whole-directory runs. File-level concurrency was already available through `-j`; the current version adds clearer process/thread reporting and a directory-capable server wrapper. Small-sample concurrency acceptance does not certify whole-directory or Gravity performance.

## Final freeze and publication

The complete regression run from the clean local 0.9.0 source directory passed: 1,626 tests passed and one optional check requiring a private real product was skipped. Public Bash examples passed syntax checks. Toy inputs were also used to exercise whole-file output, directory concurrency, resume, chunk merging, layouts, postprocessing, and plotting through the installed wheel. These checks do not establish a new clean dependency environment, Gravity scheduler acceptance, or instrument science calibration. Detailed logs remain local; a short validation summary and SHA values accompany the artifacts.

1. Confirm that source version `0.9.0`, version declarations, release notes, Gravity expected identity, and user documentation agree.
2. Rebuild wheel/sdist; verify inclusion of MIT, resource notices, and file inventories, plus the necessary compatibility entry points and self-contained tests.
3. Validate the complete regression suite and installed public examples in the stated environment. Conda installation instructions are not evidence of a completed fresh Conda installation; bind results to the actual environment and source.
4. Inspect the explicit release file list and Git diff, excluding private and historical research content. Under the user's publication authorization, commit, push, and create the version tag and GitHub Release.
5. Retain artifact SHA values, released source identity, and a validation summary for reproduction.

GitHub publication, server installation, and qsub submission have not been performed automatically by this checklist.
Gravity whole-library recalculation, instrument science calibration, and CESS+ development are separate tasks; completing all of them is not a prerequisite for publishing the one-dimensional v0.9.0 software.
