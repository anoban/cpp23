# find the files with the given keyword

$files = [System.Collections.ArrayList]::new()

Get-ChildItem -Path *.cpp | Foreach-Object {
    $str = Get-Content -Path $_.FullName
    if(Select-String -InputObject $str -Pattern "January") {
        [void]$files.Add($_.FullName)
    }
}

Write-Host $files
