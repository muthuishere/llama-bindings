# Publishing

## Versioning strategy

All three targets follow [Semantic Versioning](https://semver.org/).  
Versions are coordinated across Go, Java, and JS when the bridge ABI changes.

## Go

The Go module is `github.com/muthuishere/llama-bindings/go`.

Publish via Taskfile:
```sh
GO_VERSION=v0.1.0 task publish-go
```

This tags `go/v0.1.0` and pushes the tag. The Go module proxy picks it up
automatically.

**Checklist before publishing:**
- [ ] `go.mod` has correct module path and Go version
- [ ] All exported symbols are documented
- [ ] `task test-go` passes
- [ ] `README.md` quick-start compiles
- [ ] Semantic version bump is intentional

## Java

The Maven coordinates are `com.example.llama:llama-java`.

Publish via Taskfile:
```sh
task publish-java
```

This runs `mvn deploy`. Configure your Maven `settings.xml` with the target
repository credentials before publishing.

**Checklist before publishing:**
- [ ] `pom.xml` has correct `groupId`, `artifactId`, and `version`
- [ ] Java 21 baseline confirmed
- [ ] Source and Javadoc JARs attached
- [ ] `task test-java` passes
- [ ] Repository URL configured in `pom.xml` or `settings.xml`

## Browser JavaScript / npm

The npm package is `@llama-bindings/js-browser`.

Publish via Taskfile:
```sh
npm login   # once
task publish-js
```

**Checklist before publishing:**
- [ ] `package.json` has correct `name`, `version`, and `main`/`exports`
- [ ] `dist/` artefacts are built by `task build-wasm && task build-js`
- [ ] `dist/` is included in the `files` field
- [ ] `task test-js` passes
- [ ] `README.md` quick-start works in a browser context
- [ ] TypeScript type definitions strategy documented (if providing `.d.ts`)

## Bridge ABI compatibility

Publishing a new version with a bridge ABI change (new or modified C function
signatures) requires:

1. Updating `bridge/include/llama_bridge.h`
2. Bumping the major version on all three targets
3. Documenting the breaking change in release notes
