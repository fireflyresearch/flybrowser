# FlyBrowser Installation Guide

This guide explains all installation options and how to use FlyBrowser after installation.

## Installation Modes

FlyBrowser supports three installation modes:

### 1. Virtual Environment (Recommended) ✅

**Best for**: Most users, development, multiple Python projects

- Installs in `~/.flybrowser/venv`
- Isolated from system Python
- No conflicts with other packages
- Easy to uninstall

```bash
# Interactive installation
./install.sh

# Or specify mode explicitly
./install.sh --install-mode venv
```

### 2. System-wide Installation

**Best for**: Single-user systems, wanting `flybrowser` command globally

- Installs globally using `pip --break-system-packages`
- Requires Python 3.9+
- ⚠️ May conflict with system packages

```bash
./install.sh --install-mode system
```

### 3. User Installation

**Best for**: Shared systems without sudo access

- Installs to `~/.local`
- No sudo required
- Commands available in `~/.local/bin`

```bash
./install.sh --install-mode user

# Make sure ~/.local/bin is in your PATH
export PATH="$HOME/.local/bin:$PATH"
```

---

## Using FlyBrowser After Installation

### Virtual Environment Mode

After installing in venv mode, you have two options:

#### Option A: Using Wrapper Commands (Automatic)

The installation creates wrapper scripts in `/usr/local/bin`:

```bash
# These work anywhere without activation
flybrowser                # Interactive REPL
flybrowser-setup          # Setup wizard
flybrowser serve          # Start server
```

#### Option B: Activate the Virtual Environment

```bash
# Activate the venv
source ~/.flybrowser/venv/bin/activate

# Now you can use Python directly
python

# Or import in scripts
python your_script.py

# Deactivate when done
deactivate
```

#### Option C: Use the venv Python directly

```bash
# Run Python scripts with the venv Python
~/.flybrowser/venv/bin/python your_script.py

# Or activate in a script
#!/Users/yourusername/.flybrowser/venv/bin/python
import flybrowser
```

---

### System-wide Mode

Commands are available globally:

```bash
# Works anywhere
flybrowser                # Interactive REPL
python -m flybrowser.cli.main
```

In Python scripts:

```python
#!/usr/bin/env python3
import flybrowser

browser = flybrowser.FlyBrowser(...)
```

---

### User Mode

Ensure `~/.local/bin` is in your PATH:

```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

# Then commands work globally
flybrowser
python3 -m flybrowser.cli.main
```

---

## Using with Jupyter Notebooks

### Virtual Environment Mode

**Option 1: Register as Jupyter Kernel** (Recommended)

```bash
# Run the helper script
./setup_jupyter_kernel.sh

# Or manually:
source ~/.flybrowser/venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=flybrowser --display-name="Python (FlyBrowser)"
```

Then in Jupyter:
- Kernel → Change Kernel → Python (FlyBrowser)

**Option 2: Start Jupyter from venv**

```bash
source ~/.flybrowser/venv/bin/activate
jupyter notebook
```

**Option 3: Use in code**

```python
# In Jupyter cell
from flybrowser import FlyBrowser

browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2"
)

# Use await directly (no asyncio.run() needed in Jupyter)
await browser.start()
await browser.goto("https://example.com")
data = await browser.extract("What is the main heading?")
await browser.stop()
```

### System-wide or User Mode

Just import and use:

```python
from flybrowser import FlyBrowser

browser = FlyBrowser(...)
await browser.start()
# ... your code
await browser.stop()
```

---

## Activating the Virtual Environment

### In Terminal (for CLI use)

```bash
# Activate
source ~/.flybrowser/venv/bin/activate

# Your prompt will change to show (flybrowser)
(flybrowser) $ python
>>> import flybrowser
>>> # Works!

# Deactivate
deactivate
```

### In Python Scripts

Add shebang to use venv Python:

```python
#!/Users/yourusername/.flybrowser/venv/bin/python
# or use: #!/usr/bin/env python (if venv is activated)

import flybrowser

async def main():
    async with flybrowser.FlyBrowser(
        llm_provider="ollama",
        llm_model="llama2"
    ) as browser:
        await browser.goto("https://example.com")
        data = await browser.extract("What is the main heading?")
        print(data)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Make it executable:

```bash
chmod +x your_script.py
./your_script.py
```

### In Jupyter Notebooks

See [Jupyter section](#using-with-jupyter-notebooks) above.

---

## Common Scenarios

### Scenario 1: Developer Working on FlyBrowser

```bash
# Install in venv mode with dev extras
./install.sh --install-mode venv --dev

# Activate for development
source ~/.flybrowser/venv/bin/activate

# Make changes, test immediately
python -m pytest tests/
```

### Scenario 2: Data Scientist Using Jupyter

```bash
# Install with Jupyter support
./install.sh --install-mode venv

# Setup Jupyter kernel
./setup_jupyter_kernel.sh

# Start Jupyter
jupyter notebook

# In notebook: Kernel → Change Kernel → Python (FlyBrowser)
```

### Scenario 3: System Administrator (Server Deployment)

```bash
# Install system-wide
./install.sh --install-mode system --with-service

# Start service
sudo systemctl start flybrowser

# Check status
sudo systemctl status flybrowser
```

### Scenario 4: Shared Server User (No Sudo)

```bash
# Install to user directory
./install.sh --install-mode user

# Add to PATH (in ~/.zshrc or ~/.bashrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

# Reload shell
source ~/.zshrc

# Use commands
flybrowser
```

---

## Verifying Installation

```bash
# Check if FlyBrowser is installed
python3 -c "import flybrowser; print(flybrowser.__version__)"

# Or use the doctor command
flybrowser doctor

# Or in venv mode
source ~/.flybrowser/venv/bin/activate
python -c "import flybrowser; print(flybrowser.__version__)"
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'flybrowser'"

**Problem**: Python can't find FlyBrowser

**Solutions**:

1. **If installed in venv**: Activate it first
   ```bash
   source ~/.flybrowser/venv/bin/activate
   ```

2. **If using Jupyter**: Register the kernel
   ```bash
   ./setup_jupyter_kernel.sh
   ```

3. **If installed as user**: Check PATH
   ```bash
   echo $PATH | grep .local
   # If not there, add to ~/.zshrc:
   export PATH="$HOME/.local/bin:$PATH"
   ```

### "command not found: flybrowser"

**If venv mode**: Wrappers not in PATH or not created

```bash
# Check if wrappers exist
ls -l /usr/local/bin/flybrowser

# Or use venv directly
source ~/.flybrowser/venv/bin/activate
python -m flybrowser.cli.main
```

**If user mode**: `~/.local/bin` not in PATH

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Uninstalling

### Using the Uninstall Script (Recommended)

```bash
# Interactive uninstall (asks what to remove)
./uninstall.sh

# Remove everything without asking
./uninstall.sh --all --force

# Remove only binaries, keep config
./uninstall.sh --keep-data
```

The uninstall script automatically detects your installation mode (venv, system, or user) and removes the appropriate files.

### What Gets Removed

Depending on your installation mode:

**Virtual Environment Mode:**
- CLI wrappers in `/usr/local/bin`
- Virtual environment in `~/.flybrowser/venv`
- Data directory (if `--all` used)

**System-wide Mode:**
- FlyBrowser package from system Python
- System-wide commands

**User Mode:**
- FlyBrowser package from `~/.local`
- Commands in `~/.local/bin`

**All Modes:**
- Jupyter kernel (if installed)
- System services (if configured)
- Data and config (if `--all` used)

### Manual Uninstall

**For venv mode:**
```bash
rm -rf ~/.flybrowser
rm -f /usr/local/bin/flybrowser*
rm -rf ~/.local/share/jupyter/kernels/flybrowser
```

**For system mode:**
```bash
python3 -m pip uninstall -y flybrowser
# or with PEP 668:
python3 -m pip uninstall -y --break-system-packages flybrowser
```

**For user mode:**
```bash
python3 -m pip uninstall -y flybrowser
rm -rf ~/.local/bin/flybrowser*
rm -rf ~/.local/lib/python*/site-packages/flybrowser*
```

---

## Quick Reference

| Installation Mode | Location | Activation Required | Best For |
|------------------|----------|---------------------|----------|
| venv (default) | `~/.flybrowser/venv` | Yes (or use wrappers) | Most users |
| system | System Python | No | Single-user systems |
| user | `~/.local` | No (if PATH set) | Shared systems |

### Activation Commands

```bash
# Venv mode
source ~/.flybrowser/venv/bin/activate

# System/User mode
# No activation needed, just use commands directly
```

### Import in Python

```python
# All modes
import flybrowser
from flybrowser import FlyBrowser

# Create instance
browser = FlyBrowser(
    llm_provider="ollama",
    llm_model="llama2"
)
```

---

For more help, see:
- [Jupyter Notebooks Guide](docs/jupyter-notebooks.md)
- [Getting Started](docs/getting-started.md)
- [Documentation Index](docs/index.md)
