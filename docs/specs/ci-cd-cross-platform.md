# Spec: CI/CD Cross-Platform Build & Release

## Status
Draft

## Summary
GitHub Actions workflows for building native bridge libraries on 5 platforms, running tests, packaging for Go/Java/JS, and creating releases with all artifacts.

## Motivation
Consumers of the library should not need cmake, llama.cpp, or a C++ compiler. Pre-built native libraries must be bundled in Go (prebuilt/), Java (resources/native/), and JS (npm dist/).

## Workflows

### 1. `ci.yml` — On every push/PR
- Build bridge on: macOS x86_64, macOS arm64, Linux x86_64, Linux arm64, Windows x86_64
- Run unit tests per binding (Go, Java, JS)
- Upload bridge artifacts per platform

### 2. `release.yml` — On version tag (v*)
- Build bridge on all 5 platforms
- Copy prebuilts into Go, Java, JS package directories
- Run full test suite
- Create GitHub Release with all artifacts
- Publish: Go module tag, Maven deploy, npm publish

## Platform Matrix

| Runner | Target | Output |
|--------|--------|--------|
| macos-13 | darwin/amd64 | libllama_bridge.dylib |
| macos-14 | darwin/arm64 | libllama_bridge.dylib |
| ubuntu-latest | linux/amd64 | libllama_bridge.so |
| ubuntu-22.04 + qemu | linux/arm64 | libllama_bridge.so |
| windows-latest | windows/amd64 | llama_bridge.dll |

## Acceptance Criteria
- [ ] CI runs on push to main and on PRs
- [ ] Bridge builds on all 5 platforms
- [ ] Unit tests pass on macOS and Linux
- [ ] Release workflow creates GitHub Release with artifacts
- [ ] Go prebuilt/ populated for all platforms
- [ ] Java resources/native/ populated for all platforms
- [ ] `task test` passes after prebuilt population
