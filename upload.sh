#/bin/sh
git clean -dfx
python3 setup.py sdist
python3 setup.py bdist_wheel
twine upload dist/*
