# script to enable syntax highlighting for select languages in GNU nano from MSYS2

# this is the full path to the directory that contains all the prebundled nanorc files
[System.String] $NANORC_PATH = "C:/msys64/usr/share/nano"
[System.String] $USER_NANORC_PATH = "C:/msys64/home/Anoban/"

# there are many .nanorc files that come bundled with nano, these are the only ones we are interested in!
[System.Collections.Generic.List[System.String]] $NANORC_FILES = @(
    "asm.nanorc",
    "c.nanorc",
    "css.nanorc",
    "default.nanorc",
    "email.nanorc",
    "html.nanorc",
    "javascript.nanorc",
    "json.nanorc",
    "markdown.nanorc",
    "nanorc.nanorc",
    "python.nanorc",
    "sh.nanorc",
    "xml.nanorc",
    "yaml.nanorc"
)

# that works :)
# $NANORC_FILES | ForEach-Object { Write-Host (Get-Content -LiteralPath "$NANORC_PATH/$_") }

New-Item -Path $USER_NANORC_PATH -Name ".nanorc" -Force
# then, include the needed .nanorc files to this .nanorc file in this format
# include /usr/share/nano/name.nanorc

# let's build a string using the select .nanorc files
[System.String] $dump = ""
$NANORC_FILES | ForEach-Object { $dump += "include /usr/share/nano/$_`n" }  # in powershell `n is used for new lines not \n

Set-Content -LiteralPath "C:/msys64/home/Anoban/.nanorc" -Value $dump -Encoding ascii -Force
