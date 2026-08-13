$bmp = New-Object System.Drawing.Bitmap 64, 48
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::FromArgb(120, 120, 120))
$ms = New-Object System.IO.MemoryStream
$bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$b = [Convert]::ToBase64String($ms.ToArray())
$bmp.Dispose()

$payload = '{"image":"data:image/jpeg;base64,' + $b + '"}'
$t = Measure-Command {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:5000/api/analyze_lite' -Method Post `
    -ContentType 'application/json' -Body $payload -TimeoutSec 30
  $r.Content
}
'time(ms): ' + [int]$t.TotalMilliseconds
'---response---'
