#/bin/sh
git clean -dfx
python3 -m build
twine upload dist/*
