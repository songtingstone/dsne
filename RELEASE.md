# How to release a new version

## Upload to test PyPI

```
export VERSION=0.1.6
git checkout -b release-${VERSION}

git commit -am "Release ${VERSION}.rc0" --allow-empty
git tag ${VERSION}.rc0

make cleanall
make build
make upload-test

# Create venv and install rc version
pip install --extra-index-url=https://test.pypi.org/simple 'dsne[test]'==${VERSION}rc0
pytest --pyargs dsne/tests/

# Delete rc tag
git tag -d ${VERSION}.rc0
```

Merge branch when CI passes

## Upload to PyPI

- Update `CHANGELOG.md`
- Update `README.md` and docs as needed

```
export VERSION=0.1.6

git commit -am "Release ${VERSION}" --allow-empty
git tag ${VERSION}

make cleanall
make build
make upload-pypi
git push origin ${VERSION}
git push
```
