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
    "/DNDEBUG",
    "/EHsc",
    "/F0x10485100",
    "/favor:INTEL64",
    "/fp:strict",
    "/fpcvt:IA",
    "/GL",
    "/Gw",
    "/I./googletest/",
    "/I./googletest/include/",
    "/jumptablerdata",
    "/MP",
    "/MT",
    "/O2",
    "/Ob3",
    "/Oi",
    "/Ot",
    "/Qpar",
    "/std:c++17",
    "/TP",
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
    "/Zc:preprocessor",
    "/link /DEBUG:NONE"
)

Write-Host "cl.exe ${cfiles} ${cflags}" -ForegroundColor Cyan
cl.exe $cfiles $cflags

# If cl.exe returned 0 (True), (i.e if the compilation succeeded,)
if ($? -eq $True){
        Remove-Item "*.obj" -Force
}
