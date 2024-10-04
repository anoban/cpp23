$cfiles = [System.Collections.ArrayList]::new()
$unrecognized = [System.Collections.ArrayList]::new()

foreach ($arg in $args) {
    if (($arg -clike "*.cpp") -or ($arg -clike "*.cc")){
        [void]$cfiles.Add($arg.ToString().Replace(".\", ""))
    }
    else {
        [void]$unrecognized.Add($arg)
    }
}

if ($unrecognized.Count -ne 0) {
    Write-Error "Incompatible files passed for compilation: ${unrecognized}"
    Exit 1
}

$cflags = @(
    "./googletest/src/gtest-all.cc",
    "/arch:AVX512",
    "/diagnostics:caret",
    "/debug:none",
    "/DNDEBUG",
    "/D_NDEBUG",
    # "/DGTEST_HAS_SEH=0",
    "/EHsc",
    "/F0x10485100",
    "-fcf-protection:full",
    "/Gd",
    "/GF",
    "/GR",
    "/GS",
    "/guard:cf",
    "/Gw",
    "/fp:fast",
    "/I./googletest/",
    "/I./googletest/include/",
    "/MT",
    "/O3",
    "/Oi",
    "/Ot",
    "/QaxCORE-AVX512",
    "/Qbranches-within-32B-boundaries",
    "/Qftz",
    "/Qgather-",
    "/Qintrinsic-promote",
	"/Qimf-absolute-error:1E-10",
	"/Qimf-accuracy-bits:30/f",
	"/Qimf-accuracy-bits:60/",
	"/Qimf-accuracy-bits:70/l",
	"/Qimf-arch-consistency:false",
	"/Qimf-domain-exclusion:0",
	"/Qimf-max-error:1E-2",
	"/Qimf-use-svml:false",
	"/Qfma",
	"/Qfp-speculation:safe",
    #####"/Qipo",   #
	"/Qkeep-static-consts-",
    "/Qm64",
    "/Qopt-assume-no-loop-carried-dep=2",
    "/Qopt-dynamic-align",
    "/Qopt-multiple-gather-scatter-by-shuffles",
    "/Qopt-prefetch:5",
    "/Qopt-prefetch-distance:10000",
	"/Qpc80",
    "/Qscatter-",
    "/Qstd:c++23",
    "/Qvec",
    "/Qvec-peel-loops",
    "/Qvec-threshold:0",
    "/Qvec-with-mask",
    "/Qunroll:10000",
    "/TP",
    "/vd2", # https://learn.microsoft.com/en-us/cpp/preprocessor/vtordisp?view=msvc-170
	"/Wabi",
    "/Wall",
	"/Wcomment",
	"/Wdeprecated",
	"/Wextra-tokens",
	"/Wformat",
	"/Wformat-security",
	"/Wmain",
	"/Wno-missing-prototypes",
	"/Wno-pointer-arith",
	"/Wreturn-type",
	"/Wshadow",
	"/Wsign-compare",
	"/Wstrict-aliasing",
	"/Wuninitialized",
	"/Wunknown-pragmas",
	"/Wunused-function",
	"/Wunused-variable",
	"/Wwrite-strings",
    "/Wall",
    "/wd4514",      # removed unreferenced inline function
    "/wd4710",      # not inlined
    "/wd4711",      # selected for inline expansion
    "/wd4820",      # struct padding
    "/wd4623",
    "/wd4625",
    "/wd4626",
    "/wd4668",
    "/wd5026",
    "/wd5027",
    "/Zc:__cplusplus",
    "/Zc:char8_t",
	"/Zc:strictStrings",
    "/Zc:twoPhase"
)

Write-Host "icx.exe ${cfiles} ${cflags}" -ForegroundColor Cyan
icx.exe $cfiles $cflags
