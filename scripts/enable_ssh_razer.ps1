# Enable OpenSSH Server on Razer (run as Administrator)
# Usage: Right-click > Run with PowerShell as Admin

Write-Host "=== OpenSSH Server Setup ===" -ForegroundColor Cyan

# 1. Install OpenSSH Server if not present
$cap = Get-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
if ($cap.State -ne 'Installed') {
    Write-Host "Installing OpenSSH Server..." -ForegroundColor Yellow
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
} else {
    Write-Host "OpenSSH Server already installed." -ForegroundColor Green
}

# 2. Start sshd and set to automatic
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Write-Host "sshd service: started (auto)" -ForegroundColor Green

# 3. Open firewall port 22
$fwRule = Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue
if (-not $fwRule) {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" `
        -DisplayName "OpenSSH SSH Server (sshd)" `
        -Enabled True -Direction Inbound -Protocol TCP `
        -Action Allow -LocalPort 22
    Write-Host "Firewall rule added for port 22." -ForegroundColor Green
} else {
    Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"
    Write-Host "Firewall rule already exists - enabled." -ForegroundColor Green
}

# 4. Confirm
$svc = Get-Service sshd
Write-Host ""
Write-Host "=== Result ===" -ForegroundColor Cyan
Write-Host "sshd status: $($svc.Status)"
Write-Host "Razer IP (LAN): $(( Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '192.168.*' } | Select-Object -First 1).IPAddress)"
Write-Host "SSH user: $env:USERNAME"
Write-Host "Test from Ivan: ssh $env:USERNAME@192.168.0.105"
