# approxMLIR-IREE testing suite

The code we wrote for this course project are distributed in `python` directory and in the `https://github.com/moomoohorse321/approxMLIR` repository.

It's really hard implementation especially writing so much code in one month (look at the approxMLIR directory and you might understand why).

## Building the testing suite
You should first install `IREE` in your environment.

```bash
# create a venv
python3 -m venv venv
source venv/bin/activate
```
Refer to this [iree versions](https://iree.dev/developers/general/release-management/) to choose the right version to download.
```bash
pip install iree-base-compiler==3.3.0
pip install iree-base-runtime==3.3.0
pip install iree-tools-tf==20250320.1206
```

The binaries (`bin/replace`, `bin/merge`, `bin/approx-opt`) are pre-built (for x86 linux).

### I want to build from the source

Okay. For `bin/replace` and `bin/merge`, you can just go to bin and make them (because their source files are in them).

For `bin/approx-opt`, you need to build it from the source using this link https://github.com/moomoohorse321/approxMLIR. 

You must checkout to the `llvm2024` branch and follow the instruction in the `README.md` to build it.

All of these steps are only necessary if you want to build the binaries from the source.

## Run the test

```bash
cd python
# run the first plot (Figure 6)
python3 verification.py
# run the second plot (Figure 4)
python3 plot1.py
# run the third plot (Figure 5)
python3 plot2.py
```