# Shared Ubuntu GPU laptop: remote access setup

This guide connects two contributors to the Ubuntu laptop with an NVIDIA RTX 3060 (6 GB) for work on this repository. Both can connect simultaneously from macOS, on the same Wi-Fi or from different networks.

Use **ordinary OpenSSH over Tailscale**, with separate Linux accounts and SSH keys. Tailscale provides network connectivity; OpenSSH authenticates each Linux user. Do not enable Tailscale SSH for this procedure (`tailscale up --ssh`). No router port forwarding is needed.

Commands labeled **Ubuntu administrator** run on the GPU laptop. Commands labeled **each contributor** run on that person's Mac unless stated otherwise. Replace example usernames and IP addresses before running commands.

## 1. Check the Ubuntu host

**Ubuntu administrator**, at the laptop:

```bash
cat /etc/os-release
nvidia-smi
```

Confirm the exact Ubuntu release and record the NVIDIA driver version. If this is Ubuntu 20.04, its standard security support ended in May 2025. Plan an upgrade to a supported LTS release compatible with the robotics stack, or arrange Ubuntu Pro/ESM coverage before maintaining it as a remote host. Coordinate upgrades and reboots with both contributors.

`nvidia-smi` should display the RTX 3060. If it fails, resolve the driver installation locally before attempting GPU workloads. Do not install an arbitrary CUDA version to fix SSH: SSH needs no CUDA. The CUDA version printed by `nvidia-smi` indicates driver compatibility, not an installed CUDA toolkit.

Keep the laptop plugged in, ventilated, and connected to a reliable network. In Ubuntu's Power settings, turn off automatic suspend while plugged in. Initially leave the lid open; closing it may suspend the host. Screen locking is fine. After any reboot, verify network, SSH, and GPU availability. Disk encryption may require someone physically present to unlock the machine at boot.

## 2. Install SSH and create separate accounts

**Ubuntu administrator:**

```bash
sudo apt update
sudo apt install openssh-server git tmux curl
sudo systemctl enable --now ssh
sudo systemctl status ssh --no-pager
```

Create one account per contributor. These are example names; reuse existing personal accounts where appropriate:

```bash
sudo adduser akshay
sudo adduser teammate
```

Keep system administration on the existing administrator account. Contributors do not need sudo for ordinary development. Never share passwords, private SSH keys, or GitHub credentials.

## 3. Install Tailscale and grant teammate access

**Ubuntu administrator:**

```bash
curl -fsSL https://tailscale.com/install.sh -o /tmp/tailscale-install.sh
less /tmp/tailscale-install.sh
sudo sh /tmp/tailscale-install.sh
sudo tailscale up
```

Press `q` to leave `less`. Follow the authentication link printed by `tailscale up`, signing in with the laptop owner's Tailscale account.

```bash
tailscale ip -4
tailscale status
```

Record the host's Tailscale IPv4 address, usually beginning with `100.`.

**Each contributor:** install Tailscale for macOS from <https://tailscale.com/download/mac>, sign in with your own account, and connect it.

For two people sharing just this laptop, the owner opens the Tailscale admin console's **Machines** page, selects the Ubuntu machine, and uses **Share** to invite the teammate. The teammate accepts using their own account. Alternatively, invite the teammate into your existing team tailnet if that is how your organization manages devices. Do not share a Tailscale login.

The host's Tailscale access policy must permit the intended users to reach TCP port 22. Sharing does not override restrictive policies. Use the official sharing documentation below if access needs adjustment. Keep access limited to the people who need it.

If UFW is active on Ubuntu, add an SSH allowance for the Tailscale interface:

```bash
sudo ufw status
sudo ufw allow in on tailscale0 to any port 22 proto tcp
```

Do not enable or reset an existing firewall remotely without reviewing its rules and having local recovery access. This rule does not remove any existing LAN/public SSH allowances. Do not create router port forwards for SSH.

## 4. Create and install each person's SSH key

**Each contributor, on their Mac:**

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/so101_gpu -C "so101-gpu"
cat ~/.ssh/so101_gpu.pub
```

Choose a passphrase. If that key filename already exists, reuse it or select a new filename; do not overwrite it. Give the administrator only the single-line `.pub` content. Keep the private file `~/.ssh/so101_gpu` on your own Mac.

**Ubuntu administrator:** install each public key into its matching account. Example for `akshay`:

```bash
sudo install -d -m 700 -o akshay -g akshay /home/akshay/.ssh
sudo touch /home/akshay/.ssh/authorized_keys
sudo nano /home/akshay/.ssh/authorized_keys
```

Paste Akshay's complete public key on its own line, preserving existing keys. Save, then:

```bash
sudo chown akshay:akshay /home/akshay/.ssh/authorized_keys
sudo chmod 600 /home/akshay/.ssh/authorized_keys
```

Repeat for `teammate`, using that person's public key and replacing every `akshay` in the commands with `teammate`. These commands assume the default `/home/USERNAME` home directories created by `adduser`.

Display the host fingerprint on Ubuntu:

```bash
sudo ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

Share that fingerprint with both contributors so they can verify the first connection.

## 5. Connect from each Mac

Edit `~/.ssh/config` on each Mac and add the following once. Replace `100.x.y.z` with the real Tailscale IP. Each person uses their own Ubuntu username:

```sshconfig
Host so101-gpu
    HostName 100.x.y.z
    User akshay
    IdentityFile ~/.ssh/so101_gpu
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

```bash
chmod 600 ~/.ssh/config
ssh so101-gpu
```

Compare the first-connection fingerprint with the administrator's value before accepting it. A key passphrase prompt refers to your local key, not the Ubuntu password.

Inside the SSH session:

```bash
whoami
hostname
nvidia-smi
```

These commands now execute on Ubuntu. `exit` returns to your Mac. The same alias works from another network as long as both devices are online and Tailscale is connected. Test it using a phone hotspot before depending on access while away.

## 6. Require keys after both users have tested access

**Ubuntu administrator:** retain a working session and local console access. First verify both contributors can open a fresh key-authenticated session. Ensure any administrator who needs remote access also has a tested key.

Edit `/etc/ssh/sshd_config` and any included configuration files so the effective global settings are:

```text
PubkeyAuthentication yes
PasswordAuthentication no
ChallengeResponseAuthentication no
PermitRootLogin no
```

Inspect existing settings rather than blindly appending duplicates. OpenSSH generally uses the first obtained value, and included files or `Match` blocks can change the result. On Ubuntu 20.04, `ChallengeResponseAuthentication` is the compatible spelling for disabling keyboard-interactive authentication.

Validate before reloading:

```bash
sudo /usr/sbin/sshd -t
sudo /usr/sbin/sshd -T | grep -E 'pubkeyauthentication|passwordauthentication|challengeresponseauthentication|kbdinteractiveauthentication|permitrootlogin'
```

If syntax validation fails, fix the configuration before continuing. If there are `Match` blocks, also check effective settings for each relevant user/connection using `sshd -T -C`.

```bash
sudo systemctl reload ssh
```

Both contributors must open another fresh session successfully before the administrator closes the original session. If access fails, correct the configuration from that retained session or the local console.

## 7. Give each contributor a separate repository clone

**Each contributor, inside their Ubuntu SSH session:**

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/22by7-raikar/SO_101_semantic_manipulation.git
cd SO_101_semantic_manipulation
git config user.name "Your Name"
git config user.email "YOUR_GITHUB_EMAIL"
git switch -c your-name/your-task
```

Skip cloning if you already have a working copy. Each person's home directory gives them an independent clone, branches, environments, and output files. Avoid editing one shared checkout concurrently.

Private repository access and pushes require each contributor's own GitHub authentication and repository permission. The Mac-to-Ubuntu SSH key does not automatically authenticate Ubuntu-to-GitHub. Do not put access tokens in clone URLs or share credentials.

Use the repository's [locked environment setup](ENVIRONMENT.md). On Ubuntu, run `bash scripts/setup.sh robot` in your own clone, then `uv run --no-sync so101-doctor --gpu --hardware`. Create Linux environments on Ubuntu, not by copying macOS virtual environments. Install dependencies per user, without `sudo pip`.

For GPU-enabled PyTorch code, after the project's chosen PyTorch build is installed, check it with:

```bash
uv run --no-sync python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

A working driver does not guarantee that a particular Python environment has GPU-enabled libraries.

### Moving code between Mac and Ubuntu

Commit and push your branch from the machine where you edited it. In the other clone, fetch and switch to that branch, then pull when appropriate. Check `git status` first and preserve local changes. Uncommitted Mac edits do not automatically appear on Ubuntu.

Both contributors should push their own branches and use pull requests to combine changes. Keep datasets, checkpoints, credentials, and large run outputs out of ordinary Git commits.

## 8. Keep jobs running and share the GPU

**Inside Ubuntu SSH, as your own user:**

```bash
tmux new -s so101
cd ~/work/SO_101_semantic_manipulation
# Activate your project environment and run your command here.
```

Detach with **Ctrl+B**, then **D**. Later:

```bash
ssh so101-gpu
tmux ls
tmux attach -t so101
```

Each Linux user can use the session name `so101` independently. tmux keeps jobs alive through SSH disconnections, but not a reboot, shutdown, or suspend. Save checkpoints for long runs.

Both users can edit and run light CPU tasks simultaneously. The RTX 3060's 6 GB VRAM is shared; SSH accounts do not partition GPU memory. Coordinate one heavy GPU job at a time initially. Before starting a job:

```bash
nvidia-smi
```

Use `watch -n 2 nvidia-smi` for monitoring (Ctrl+C stops the monitor). Agree on GPU time slots and ask the job owner before stopping anything. Do not kill another person's processes or change drivers while they are working. `CUDA_VISIBLE_DEVICES=0` selects the GPU; it does not reserve it.

## 9. Camera, robot, and graphical tools

The RealSense camera and SO-101 hardware must be connected to the Ubuntu machine running the hardware process. SSH does not forward USB devices from the Mac. Configure device permissions/udev rules using the project's eventual hardware setup instructions.

Only one person should own robot control at a time. Coordinate physical operation with someone at the robot and the project's hardware safety procedure.

Ordinary SSH does not display a remote MuJoCo or camera GUI on the Mac. Use headless processing and saved artifacts where supported; arrange remote desktop separately if interactive graphics are required.

For a development dashboard, bind the service on Ubuntu to `127.0.0.1` and forward its port. Example, if your service uses port 8888, run on the Mac:

```bash
ssh -N -L 8888:127.0.0.1:8888 so101-gpu
```

Open `http://localhost:8888` on your Mac. Keep the service's authentication enabled. Use different remote ports if both people start servers. This tunnel does not start the server itself.

## 10. Acceptance checklist and troubleshooting

- [ ] Each person has their own Tailscale identity, Linux account, and SSH key.
- [ ] Both can connect concurrently and `whoami` shows different users.
- [ ] Both can run `nvidia-smi`.
- [ ] Both can reconnect to their own tmux session after disconnecting.
- [ ] SSH works from a different network, such as a phone hotspot.
- [ ] Laptop remains reachable while plugged in and idle.
- [ ] Key-only access works in fresh sessions after SSH configuration changes.
- [ ] Each person has a separate clone and an agreed GPU/robot schedule.

| Symptom | Check |
| --- | --- |
| Connection times out | Host awake and online; both Tailscale clients connected; sharing accepted; policy permits TCP 22; firewall rules. |
| Connection refused | On Ubuntu, inspect `sudo systemctl status ssh` and `sudo ss -ltnp`. |
| Permission denied (publickey) | Correct Ubuntu username, matching public key, ownership and permissions; run `ssh -v so101-gpu` on the Mac. |
| Host key changed | Verify with the administrator before changing `known_hosts`; an OS reinstall can change keys. |
| Tailscale host disappears | Check host power/network and device authentication/key expiry in the admin console. |
| NVIDIA driver error | Check locally with `nvidia-smi`; coordinate driver repairs and any reboot. |
| CUDA out of memory | Check the other user's jobs; reduce workload/batch size or wait for the GPU. |
| Job stopped after reboot | tmux cannot survive reboot; resume from a saved checkpoint. |

To revoke a teammate's access, remove their Tailscale share/team access and their SSH public keys on Ubuntu. Removing keys does not terminate existing sessions or running jobs; the administrator should review and end those as appropriate. Preserve their work before disabling or removing their account.

## Official references

- [Ubuntu OpenSSH server](https://ubuntu.com/server/docs/how-to/security/openssh-server/)
- [Install Tailscale on Linux](https://tailscale.com/docs/install/linux)
- [Share a Tailscale machine with another user](https://tailscale.com/docs/features/sharing)
- [Tailscale: inviting users versus sharing a device](https://tailscale.com/docs/reference/inviting-vs-sharing)
- [Ubuntu 20.04 support status](https://ubuntu.com/20-04)

This is a setup guide, not a record that these commands have already been run on the Ubuntu laptop.
