# Chrome DevTools MCP Server

An MCP (Model Context Protocol) server that connects to Chrome browser's DevTools Protocol, enabling AI assistants to interact with web pages, inspect DOM elements, execute JavaScript, and more.

## Features

- 🔌 **Connect to Chrome**: Connect to Chrome browser running with remote debugging
- 🔍 **Inspect Elements**: Query and inspect DOM elements by CSS selector
- 📜 **Execute JavaScript**: Run JavaScript code in the page context
- 📊 **Get DOM Tree**: Retrieve the DOM structure of the current page
- 📸 **Take Screenshots**: Capture screenshots of the current page
- 🌐 **Network Monitoring**: Get network requests made by the page
- 📝 **Console Logs**: Access console messages from the page
- 🧭 **Navigation**: Navigate to URLs programmatically

## Prerequisites

1. **Python 3.8+**
2. **Chrome Browser** with remote debugging enabled
3. **MCP Client** (e.g., Claude Desktop, Cursor, etc.)

## Installation

1. **Install dependencies**:
   ```bash
   # Install base dependencies
   pip install -r requirements.txt
   
   # Or use the installation script
   ./install_mcp.sh
   ```

2. **Install MCP Python SDK**:
   The MCP package may need to be installed separately. Try:
   ```bash
   pip install mcp
   ```
   
   If that doesn't work, check the official MCP Python SDK:
   - Visit: https://github.com/modelcontextprotocol/python-sdk
   - Follow their installation instructions
   - Or install from source if needed

2. **Start Chrome with remote debugging**:
   ```bash
   # macOS
   /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

   # Linux
   google-chrome --remote-debugging-port=9222

   # Windows
   chrome.exe --remote-debugging-port=9222
   ```

   Or use an existing Chrome instance by adding the flag when launching.

## Configuration

### For Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "python",
      "args": [
        "/absolute/path/to/mcp_chrome_devtools_server.py"
      ],
      "env": {
        "CHROME_DEBUG_PORT": "9222"
      }
    }
  }
}
```

**Important**: Update the path to `mcp_chrome_devtools_server.py` with your actual absolute path.

### For Cursor

Add to your Cursor settings or MCP configuration:

```json
{
  "mcpServers": {
    "chrome-devtools": {
      "command": "python",
      "args": [
        "/absolute/path/to/mcp_chrome_devtools_server.py"
      ]
    }
  }
}
```

## Usage

Once configured, you can use the MCP server through your AI assistant. Here are some example commands:

### Connect to Chrome
```
Connect to Chrome browser on port 9222
```

### Inspect an Element
```
Inspect the element with selector "button.submit-btn"
```

### Get DOM Tree
```
Get the DOM tree structure of the current page
```

### Execute JavaScript
```
Execute JavaScript: document.querySelector('h1').textContent
```

### Take Screenshot
```
Take a screenshot of the current page
```

### Navigate to URL
```
Navigate to https://example.com
```

### Get Page Info
```
Get information about the current page
```

## Available Tools

1. **connect_chrome**: Connect to Chrome browser
   - `port` (optional): Chrome remote debugging port (default: 9222)
   - `target_url` (optional): URL of the target page to connect to

2. **inspect_element**: Inspect a DOM element
   - `selector` (required): CSS selector for the element

3. **get_dom_tree**: Get DOM tree structure
   - `depth` (optional): Depth of the DOM tree (-1 for full tree, default: 3)

4. **execute_javascript**: Execute JavaScript code
   - `code` (required): JavaScript code to execute

5. **get_console_logs**: Get console messages

6. **get_network_requests**: Get network requests

7. **take_screenshot**: Take a screenshot

8. **navigate**: Navigate to a URL
   - `url` (required): URL to navigate to

9. **get_page_info**: Get page information (title, URL, etc.)

## Troubleshooting

### Chrome Connection Issues

If you get connection errors:

1. **Verify Chrome is running with remote debugging**:
   ```bash
   curl http://localhost:9222/json
   ```
   This should return a JSON array of available targets.

2. **Check the port**: Make sure the port in your configuration matches the port Chrome is using.

3. **Firewall**: Ensure your firewall isn't blocking the connection.

### Python Dependencies

If you encounter import errors:

```bash
# Install required packages
pip install websockets httpx

# Install MCP package (try different options)
pip install mcp
# OR
pip install @modelcontextprotocol/sdk-python
# OR check: https://github.com/modelcontextprotocol/python-sdk
```

You can also run the test script to verify your setup:
```bash
python test_mcp_server.py
```

### Multiple Chrome Instances

If you have multiple Chrome instances, the server will connect to the first available target. You can specify a `target_url` when connecting to target a specific page.

## Development

The MCP server uses the Chrome DevTools Protocol (CDP) to communicate with Chrome. It:

1. Connects to Chrome's remote debugging endpoint
2. Discovers available targets (tabs/pages)
3. Establishes a WebSocket connection to the target
4. Sends CDP commands and receives responses

## Security Notes

⚠️ **Warning**: Running Chrome with remote debugging enabled exposes debugging capabilities. Only use this in a trusted environment and avoid exposing the debugging port to the network.

To limit access to localhost only (default behavior):
```bash
chrome --remote-debugging-port=9222 --remote-debugging-address=127.0.0.1
```

## License

This MCP server is part of the Intelli-ODM project.

## Contributing

Feel free to extend this server with additional Chrome DevTools Protocol capabilities as needed.

