# Convert legacy .doc/.rtf files in data\word to .docx (via Word) into data\word_converted. Resumable.
param([string]$Src = "$PSScriptRoot\..\data\word", [string]$Dst = "$PSScriptRoot\..\data\word_converted",
      [int]$Shard = 0, [int]$Shards = 1)   # run several copies with -Shard 0..N-1 -Shards N to convert in parallel

$Src = (Resolve-Path $Src).Path
New-Item -ItemType Directory -Force $Dst | Out-Null
$Dst = (Resolve-Path $Dst).Path
$log = Join-Path $Dst "_failures_$Shard.txt"

$all = @(Get-ChildItem -LiteralPath $Src -File | Where-Object { $_.Extension -match '^\.(doc|rtf)$' } | Sort-Object Name)
$files = for ($i = 0; $i -lt $all.Count; $i++) { if ($i % $Shards -eq $Shard) { $all[$i] } }
$word = $null
function Start-Word {
    $w = New-Object -ComObject Word.Application
    $w.Visible = $false; $w.DisplayAlerts = 0; $w.AutomationSecurity = 3
    return $w
}

$done = 0; $failed = 0
foreach ($f in $files) {
    $out = Join-Path $Dst ($f.BaseName + ".docx")
    if (Test-Path -LiteralPath $out) { continue }
    if ($f.Length -eq 0) { continue }

    # Some ".doc" files are really .docx (zip) containers: just copy them.
    $fs = [System.IO.File]::OpenRead($f.FullName); $b = New-Object byte[] 2; [void]$fs.Read($b, 0, 2); $fs.Close()
    if ($b[0] -eq 0x50 -and $b[1] -eq 0x4B) { Copy-Item -LiteralPath $f.FullName -Destination $out; $done++; continue }

    try {
        if ($null -eq $word) { $word = Start-Word }
        # Open(FileName, ConfirmConversions, ReadOnly, AddToRecentFiles, PasswordDocument) - dummy password avoids prompts
        $doc = $word.Documents.Open($f.FullName, $false, $true, $false, "x")
        $target = [string]$out; $format = [int]16   # plain types: PowerShell-wrapped values break COM binding
        $doc.SaveAs([ref]$target, [ref]$format)
        $doc.Close($false)
        $done++
    } catch {
        $failed++
        Add-Content -LiteralPath $log -Value "$($f.Name)`t$($_.Exception.Message)" -Encoding utf8
        try { $word.Quit() } catch {}
        $word = $null
    }
    if (($done + $failed) % 100 -eq 0) { Write-Output "converted $done, failed $failed" }
}
if ($word) { $word.Quit() }
Write-Output "DONE converted $done, failed $failed, of $($files.Count)"
