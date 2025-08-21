from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from textual.app import App, ComposeResult
from textual.widgets import Static, RichLog, Button, Input, Label
from textual.containers import Horizontal, HorizontalGroup, Container, ScrollableContainer
from textual.screen import ModalScreen
from textual import work, on
from textual.reactive import reactive
from textual.logging import TextualHandler
import logging
import asyncio
from art import *
from mythologizer_postgres.db import ping_db_basic
from mythologizer_postgres.connectors import get_simulation_status
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import subprocess
import sys
import signal

# Configure logging to use TextualHandler
logging.basicConfig(
    level="DEBUG",
    handlers=[TextualHandler()],
)


# Configuration file handling
CONFIG_FILE = Path("simulation_config.json")

def load_simulation_config() -> Dict[str, Any]:
    """Load simulation configuration from JSON file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Return default configuration
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "agent_attributes_file": "agent_attributes.py",
        "mythemes_file": "mythemes.txt",
        "initial_cultures_file": "",
        "epochs": ""
    }

def save_simulation_config(config: Dict[str, Any]) -> None:
    """Save simulation configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

class Header(Static):
    """Header widget."""
    def compose(self) -> ComposeResult:
        asci_art = text2art("Mythologizer", font="Caligraphy2")
        yield Static(asci_art, classes="header-title")
        

class SecondaryHeader(Horizontal):
    """Secondary header widget."""
    
    is_connected = reactive(False)
    current_epoch = reactive(0)
    n_agents = reactive(0)
    n_myths = reactive(0)
    n_cultures = reactive(0)
    
    def compose(self) -> ComposeResult:
        with HorizontalGroup(classes="status-wrapper"):
            yield Static("Checking connection...", classes="status-item", id="db_status")
            yield Static("|", classes="separator")
        with HorizontalGroup(classes="status-wrapper"):
            yield Static("Current Epoch: 0", id="epoch_display", classes="status-item")
            yield Static("|", classes="separator")
        with HorizontalGroup(classes="status-wrapper"):
            yield Static("Number of Agents: 0", id="agents_display", classes="status-item")
            yield Static("|", classes="separator")
        with HorizontalGroup(classes="status-wrapper"):
            yield Static("Number of Cultures: 0", id="cultures_display", classes="status-item")
            yield Static("|", classes="separator")
        with HorizontalGroup(classes="status-wrapper"):
            yield Static("Number of Myths: 0", id="myths_display", classes="status-item")

            


    def on_mount(self) -> None:
        """Start the database monitoring when the widget is mounted."""
        self.start_db_monitoring()
    
    @work
    async def start_db_monitoring(self) -> None:
        """Start monitoring database connectivity in the background."""
        while True:
            try:
                # Run database ping in a separate thread to avoid blocking
                connection_status = await asyncio.to_thread(ping_db_basic)
                self.is_connected = connection_status is not False
            except Exception as e:
                # Any error - just mark as disconnected
                self.is_connected = False
            
            await self.update_status_display()
            
            # Only update simulation status if database is connected
            if self.is_connected:
                try:
                    await self.update_simulation_status()
                except Exception:
                    # If simulation status update fails, just continue
                    pass
            
            await asyncio.sleep(1)
    
    async def update_status_display(self) -> None:
        """Update the status display based on connection state."""
        status_widget = self.query_one("#db_status")
        
        if self.is_connected:
            status_widget.update("Connected")
            # Enable the simulation buttons
            self.update_simulation_buttons(True)
        else:
            status_widget.update("Disconnected")
            # Disable the simulation buttons
            self.update_simulation_buttons(False)
            # Clear simulation status when disconnected
            self.clear_simulation_status()
    
    def update_simulation_buttons(self, enabled: bool) -> None:
        """Update the simulation buttons based on database connection."""
        try:
            setup_btn = self.app.query_one("#setup_btn")
            run_btn = self.app.query_one("#run_btn")
            
            if enabled:
                setup_btn.disabled = False
                setup_btn.tooltip = None
                run_btn.disabled = False
                run_btn.tooltip = None
            else:
                setup_btn.disabled = True
                setup_btn.tooltip = "Database must be connected"
                run_btn.disabled = True
                run_btn.tooltip = "Database must be connected"
        except Exception:
            # If buttons don't exist yet, ignore the error
            pass
    
    async def update_simulation_status(self) -> None:
        """Update simulation status when database is connected."""
        try:
            # Get simulation status from the database in a separate thread
            status = await asyncio.to_thread(get_simulation_status)
            
            # Update the reactive variables
            self.current_epoch = status['current_epoch']
            self.n_agents = status['n_agents']
            self.n_myths = status['n_myths']
            self.n_cultures = status['n_cultures']
            
            # Update the display widgets
            self.query_one("#epoch_display").update(f"Current Epoch: {status['current_epoch']}")
            self.query_one("#agents_display").update(f"Number of Agents: {status['n_agents']}")
            self.query_one("#myths_display").update(f"Number of Myths: {status['n_myths']}")
            self.query_one("#cultures_display").update(f"Number of Cultures: {status['n_cultures']}")
        except Exception as e:
            # If there's an error, keep the last known values and don't crash
            # This prevents the app from freezing when database connection fails
            pass
    
    def clear_simulation_status(self) -> None:
        """Clear simulation status display when disconnected."""
        try:
            # Update the display widgets to show "-" when disconnected
            self.query_one("#epoch_display").update("Current Epoch: -")
            self.query_one("#agents_display").update("Number of Agents: -")
            self.query_one("#myths_display").update("Number of Myths: -")
            self.query_one("#cultures_display").update("Number of Cultures: -")
        except Exception:
            # If widgets don't exist yet, ignore the error
            pass

class Body(Static):
    """Body widget."""
    
    def __init__(self):
        super().__init__()
        self.showing_settings = False
    
    def compose(self) -> ComposeResult:
        yield Menu()



class SimulationLogView(Static):
    """Log view for simulation operations."""
    
    def compose(self) -> ComposeResult:
        yield RichLog(id="simulation_log", auto_scroll=True, highlight=False, max_lines=500)    

    

    
    @on(Button.Pressed, "#back_to_menu")
    def back_to_menu(self) -> None:
        """Go back to the main menu."""
        body = self.parent  # SimulationLogView -> Body
        body.remove_children()
        body.mount(Menu())
    


class SetupConfirmationModal(ModalScreen[bool]):
    """Confirmation modal for setup simulation."""
    
    DEFAULT_CSS = """
    SetupConfirmationModal {
        align: center middle;
    }
    
    SetupConfirmationModal > Container {
        width: auto;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }
    
    SetupConfirmationModal > Container > Label {
        width: 100%;
        content-align-horizontal: center;
        margin-top: 1;
    }
    
    SetupConfirmationModal > Container > Horizontal {
        width: auto;
        height: auto;
    }
    
    SetupConfirmationModal > Container > Horizontal > Button {
        margin: 2 4;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Label("⚠️  Setting up a simulation will delete the previous one.\nAre you sure you want to continue?")
            with Horizontal():
                yield Button("Cancel", id="cancel_setup", variant="default")
                yield Button("Continue", id="confirm_setup", variant="error")
    
    @on(Button.Pressed, "#cancel_setup")
    def cancel_setup(self) -> None:
        """Cancel the setup."""
        self.dismiss(False)
    
    @on(Button.Pressed, "#confirm_setup")
    def confirm_setup(self) -> None:
        """Confirm the setup."""
        self.dismiss(True)

class Menu(Static):
    """Menu widget."""
    
    def compose(self) -> ComposeResult:
        yield Button("Settings", id="settings_btn", classes="menu-item")
        yield Button("Setup Simulation", id="setup_btn", classes="menu-item", disabled=True, tooltip="Database must be connected")
        yield Button("Run Simulation", id="run_btn", classes="menu-item", disabled=True, tooltip="Database must be connected")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "settings_btn":
            # Get the body widget and switch its content
            body = self.parent
            body.showing_settings = True
            body.remove_class("menu-view")
            body.add_class("settings-view")
            
            # Clear and replace content
            body.remove_children()
            body.mount(SettingsView())
        elif event.button.id == "setup_btn":
            # Show confirmation modal
            self.app.push_screen(SetupConfirmationModal(), self.setup_confirmation_callback)
        elif event.button.id == "run_btn":
            # Start run simulation directly
            self.start_run_simulation()
    
    def setup_confirmation_callback(self, confirmed: bool) -> None:
        """Handle the setup confirmation result."""
        if confirmed:
            # Switch to log view
            body = self.app.query_one("Body")
            body.remove_children()
            log_view = SimulationLogView()
            body.mount(log_view)
            
            # Start setup simulation in background using the app's worker
            self.app.call_after_refresh(self.app.run_setup_simulation, log_view)
        else:
            self.app.notify("Setup cancelled", severity="information")
    
    def start_run_simulation(self) -> None:
        """Start the run simulation."""
        # Switch to log view
        body = self.app.query_one("Body")
        body.remove_children()
        log_view = SimulationLogView()
        body.mount(log_view)
        
        # Start run simulation in background using the app's worker
        self.app.call_after_refresh(self.app.run_simulation, log_view)
    


class SettingsView(ScrollableContainer):
    """Settings view widget."""
    def compose(self) -> ComposeResult:
        with Container(classes="settings-container"):
            yield DbSettings()
            yield SetupSettings()
            yield SettingsButtons()


class SettingsButtons(Static):
    """Settings button widget."""
    def compose(self) -> ComposeResult:
        yield Button("Save", id="save_btn", classes="settings-buttons")
        yield Button("Cancel", id="cancel_btn", classes="settings-buttons")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel_btn":
            self.go_back_to_menu()
        elif event.button.id == "save_btn":
            self.save_settings()
    
    def go_back_to_menu(self) -> None:
        """Go back to the main menu."""
        # Get the body widget and switch back to menu
        body = self.parent.parent  # SettingsView -> Body
        body.showing_settings = False
        body.remove_class("settings-view")
        body.add_class("menu-view")
        
        # Clear and replace content with menu
        body.remove_children()
        body.mount(Menu())
    
    def save_settings(self) -> None:
        """Save settings to .env file and simulation config with validation."""
        try:
            from dotenv import set_key
            
            # Get the SettingsView widget to access the input fields
            settings_view = self.parent  # SettingsButtons -> SettingsView
            db_settings = settings_view.query_one(DbSettings)
            setup_settings = settings_view.query_one(SetupSettings)
            
            # Get database values from input fields
            db_host = db_settings.query_one("#db_host").value
            db_port = db_settings.query_one("#db_port").value
            db_user = db_settings.query_one("#db_user").value
            db_password = db_settings.query_one("#db_password").value
            db_name = db_settings.query_one("#db_name").value
            
            # Get simulation settings values
            embedding_model = setup_settings.query_one("#embedding_model").value
            agent_attributes_file = setup_settings.query_one("#agent_attributes_file").value
            mythemes_file = setup_settings.query_one("#mythemes_file").value
            initial_cultures_file = setup_settings.query_one("#initial_cultures_file").value
            epochs = setup_settings.query_one("#epochs").value
            
            # Validate that required database fields are not empty
            if not db_host or not db_user or not db_name:
                self.app.notify("Please fill in all required database fields (Host, User, Database)", severity="warning")
                return
            
            # Validate that required simulation fields are not empty
            if not embedding_model or not agent_attributes_file or not mythemes_file:
                self.app.notify("Please fill in all required simulation fields", severity="warning")
                return
            
            # Save database settings to .env file
            set_key(".env", "POSTGRES_HOST", db_host)
            set_key(".env", "POSTGRES_PORT", db_port)
            set_key(".env", "POSTGRES_USER", db_user)
            set_key(".env", "POSTGRES_PASSWORD", db_password)
            set_key(".env", "POSTGRES_DB", db_name)
            
            # Save simulation settings to JSON config file
            simulation_config = {
                "embedding_model": embedding_model,
                "agent_attributes_file": agent_attributes_file,
                "mythemes_file": mythemes_file,
                "initial_cultures_file": initial_cultures_file,
                "epochs": epochs
            }
            save_simulation_config(simulation_config)
            
            # Show success message
            self.app.notify("Settings saved successfully!", severity="information")
            
            # Reload environment variables so database functions pick up new settings
            load_dotenv(override=True)
            
            # Reset connection status to force a fresh check with new settings
            self.reset_database_connection()
            
            # Go back to menu
            self.go_back_to_menu()
            
        except Exception as e:
            self.app.notify(f"Error saving settings: {str(e)}", severity="error")
    
    def reset_database_connection(self) -> None:
        """Reset database connection status to force a fresh check."""
        try:
            # Get the secondary header and reset its connection status
            secondary_header = self.app.query_one(SecondaryHeader)
            if secondary_header:
                # Reset connection status to force a fresh check
                secondary_header.is_connected = False
                
                # Force an immediate database check with new settings
                self.force_immediate_db_check(secondary_header)
                
        except Exception:
            # If we can't find the secondary header, just continue
            pass
    
    def force_immediate_db_check(self, secondary_header) -> None:
        """Force an immediate database check with new settings."""
        try:
            # Clear any cached environment variables
            import os
            # Remove the database-related environment variables to force fresh load
            for key in ['POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']:
                if key in os.environ:
                    del os.environ[key]
            
            # Reload environment variables
            load_dotenv(override=True)
            
            # Force an immediate connection test
            try:
                connection_status = ping_db_basic()
                secondary_header.is_connected = connection_status is not False
            except Exception:
                secondary_header.is_connected = False
            
            # Update the display immediately
            secondary_header.app.call_after_refresh(secondary_header.update_status_display)
            
        except Exception:
            # If anything fails, just mark as disconnected
            secondary_header.is_connected = False

class SetupSettings(Static):
    """Setup settings widget."""
    def compose(self) -> ComposeResult:
        # Load current configuration
        config = load_simulation_config()
        
        yield Label("Sentence Embedding Model", classes="input-label")
        yield Input(value=config.get("embedding_model", "all-MiniLM-L6-v2"), id="embedding_model", classes="input-field")
        yield Label("Agent Attributes File", classes="input-label")
        yield Input(value=config.get("agent_attributes_file", "agent_attributes.py"), id="agent_attributes_file", classes="input-field")
        yield Label("Mythemes File", classes="input-label")
        yield Input(value=config.get("mythemes_file", "mythemes.txt"), id="mythemes_file", classes="input-field")
        yield Label("Initial Cultures File (optional)", classes="input-label")
        yield Input(value=config.get("initial_cultures_file", ""), id="initial_cultures_file", classes="input-field")
        yield Label("Number of Epochs (optional)", classes="input-label")
        yield Input(value=config.get("epochs", ""), id="epochs", classes="input-field")


class DbSettings(Static):
    """Database settings widget."""
    
    def __init__(self):
        super().__init__()
        self.load_env_values()
    
    def load_env_values(self) -> None:
        """Load values from .env file."""
        # Load .env file if it exists
        load_dotenv()
        
        # Get values from environment variables with defaults
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_PORT", "5432")
        self.db_user = os.getenv("POSTGRES_USER", "")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "")
        self.db_name = os.getenv("POSTGRES_DB", "")
    
    def compose(self) -> ComposeResult:
        yield Label("Postgres Host", classes="input-label")
        yield Input(value=self.db_host, id="db_host", classes="input-field")
        yield Label("Postgres Port", classes="input-label")
        yield Input(value=self.db_port, id="db_port", classes="input-field")
        yield Label("Postgres User", classes="input-label")
        yield Input(value=self.db_user, id="db_user", classes="input-field")
        yield Label("Postgres Password", classes="input-label")
        yield Input(value=self.db_password, id="db_password", classes="input-field", password=True)
        yield Label("Postgres Database", classes="input-label")
        yield Input(value=self.db_name, id="db_name", classes="input-field")

class Footer(Static):
    """Footer widget."""
    def compose(self) -> ComposeResult:
        text = "Visit [bold]https://mythologizer.org[/bold] to learn more aswell as to create your own agents and browse though past myths."
        yield Static(text, classes="footer-text", markup=True)

    

class MythologizerApp(App):
    """Simple Hello World Textual app."""
    CSS_PATH = "app.tcss"
    
    def __init__(self):
        super().__init__()
        self._shutdown_requested = False
    
    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()
        yield SecondaryHeader()
        yield Body()
        yield Footer()
    
    def on_unmount(self) -> None:
        """Clean up when the app is unmounted."""
        self._shutdown_requested = True
    
    def exit(self, result=None) -> None:
        """Override exit to ensure proper cleanup."""
        self._shutdown_requested = True
        super().exit(result)
    
    def on_key(self, event) -> None:
        """Handle keyboard events."""
        if event.key == "ctrl+q":
            self._shutdown_requested = True
            self.exit()
    
    def is_shutting_down(self) -> bool:
        """Check if the app is shutting down."""
        return self._shutdown_requested or not self.is_running
    
    @work
    async def run_setup_simulation(self, log_view: SimulationLogView) -> None:
        """Run setup simulation in background with logging."""
        # Get the RichLog widget
        log_widget = log_view.query_one("#simulation_log")
        
        # Create custom TextualHandler to redirect logs to our widget
        class TextualHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Use call_after_refresh to ensure UI thread safety
                    self.app.call_after_refresh(lambda: log_widget.write(msg))
                except Exception:
                    self.handleError(record)
        
        # Set up the handler
        handler = TextualHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handler.app = self  # Store app reference
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        
        try:
            # Load the saved configuration
            config = load_simulation_config()
            logging.info(f"Loaded configuration: {config}")
            
            # Import the setup function
            from mythologizer_core.setup import setup_simulation
            logging.info("Imported setup_simulation function")
            
            # Load agent attributes from the specified file
            import importlib.util
            
            logging.info(f"Loading agent attributes from: {config['agent_attributes_file']}")
            
            # Check if agent attributes file exists
            agent_attributes_file = config["agent_attributes_file"]
            if not os.path.exists(agent_attributes_file):
                raise FileNotFoundError(f"Agent attributes file not found: {agent_attributes_file}")
            
            # Check if mythemes file exists
            mythemes_file = config["mythemes_file"]
            if not os.path.exists(mythemes_file):
                raise FileNotFoundError(f"Mythemes file not found: {mythemes_file}")
            
            logging.info(f"Files exist: {agent_attributes_file} and {mythemes_file}")
            
            # Load the agent attributes file
            spec = importlib.util.spec_from_file_location("agent_attributes_module", agent_attributes_file)
            agent_attributes_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_attributes_module)
            
            # Get the agent_attributes list from the module
            agent_attributes = getattr(agent_attributes_module, "agent_attributes")
            logging.info(f"Loaded {len(agent_attributes)} agent attributes")
            
            # Call setup_simulation with the saved configuration
            logging.info("Calling setup_simulation...")
            logging.info(f"Parameters: embedding_model={config['embedding_model']}, mythemes_file={config['mythemes_file']}, agent_attributes_count={len(agent_attributes)}")
            logging.info(f"Agent attributes type: {type(agent_attributes)}")
            logging.info(f"Agent attributes: {agent_attributes}")
            
            # Get initial cultures file (optional)
            initial_cultures_file = config.get("initial_cultures_file", "")
            if initial_cultures_file:
                logging.info(f"Initial cultures file: {initial_cultures_file}")
            else:
                logging.info("No initial cultures file specified")
            
            # Run setup_simulation in background thread with fresh connection
            # Run the setup simulation directly
            result = await asyncio.to_thread(
                setup_simulation,
                embedding_function=config["embedding_model"],
                mythemes=config["mythemes_file"],
                agent_attributes=agent_attributes,
                inital_cultures=initial_cultures_file if initial_cultures_file else None
            )
            
            logging.info("Setup simulation completed successfully!")
            self.call_after_refresh(lambda: self.notify("Setup simulation completed successfully!", severity="information"))
            
            # Return to menu after successful setup
            self.call_after_refresh(self.return_to_menu_after_setup)
            
        except Exception as e:
            error_msg = f"Setup failed: {str(e)}"
            logging.error(f"Setup failed: {str(e)}")
            self.call_after_refresh(lambda: self.notify(error_msg, severity="error"))
            
            # Return to menu after failed setup as well
            self.call_after_refresh(self.return_to_menu_after_setup)
        finally:
            # Clean up: remove our custom handler
            root_logger.removeHandler(handler)
    
    @work
    async def run_simulation(self, log_view: SimulationLogView) -> None:
        """Run simulation in background with logging."""
        # Get the RichLog widget
        log_widget = log_view.query_one("#simulation_log")
        
        # Create custom TextualHandler to redirect logs to our widget
        class TextualHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Use call_after_refresh to ensure UI thread safety
                    self.app.call_after_refresh(lambda: log_widget.write(msg))
                except Exception:
                    self.handleError(record)
        
        # Set up the handler
        handler = TextualHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handler.app = self  # Store app reference
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)
        
        try:
            # Load the saved configuration
            config = load_simulation_config()
            logging.info(f"Loaded configuration: {config}")
            
            # Import the run function
            from mythologizer_core import run_simulation
            logging.info("Imported run_simulation function")
            
            # Get epochs value (optional)
            epochs = config.get("epochs", "")
            if epochs:
                try:
                    epochs = int(epochs)
                    logging.info(f"Running simulation for {epochs} epochs")
                except ValueError:
                    logging.warning(f"Invalid epochs value '{epochs}', running without epoch limit")
                    epochs = None
            else:
                logging.info("No epochs specified, running without epoch limit")
                epochs = None
            
            # Load agent attributes from the specified file
            import importlib.util
            
            logging.info(f"Loading agent attributes from: {config['agent_attributes_file']}")
            
            # Check if agent attributes file exists
            agent_attributes_file = config["agent_attributes_file"]
            if not os.path.exists(agent_attributes_file):
                raise FileNotFoundError(f"Agent attributes file not found: {agent_attributes_file}")
            
            logging.info(f"Agent attributes file exists: {agent_attributes_file}")
            
            # Load the agent attributes file
            spec = importlib.util.spec_from_file_location("agent_attributes_module", agent_attributes_file)
            agent_attributes_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_attributes_module)
            
            # Get the agent_attributes list from the module
            agent_attributes = getattr(agent_attributes_module, "agent_attributes")
            logging.info(f"Loaded {len(agent_attributes)} agent attributes")
            
            # Call run_simulation
            logging.info("Calling run_simulation...")
            result = await asyncio.to_thread(
                run_simulation, 
                agent_attributes=agent_attributes, 
                n_epochs=epochs,
                should_cancel=self.is_shutting_down
            )
            
            logging.info("Simulation completed successfully!")
            self.call_after_refresh(lambda: self.notify("Simulation completed successfully!", severity="information"))
            
            # Return to menu after successful simulation
            self.call_after_refresh(self.return_to_menu_after_simulation)
            
        except Exception as e:
            error_msg = f"Simulation failed: {str(e)}"
            logging.error(f"Simulation failed: {str(e)}")
            self.call_after_refresh(lambda: self.notify(error_msg, severity="error"))
            
            # Return to menu after failed simulation as well
            self.call_after_refresh(self.return_to_menu_after_simulation)
        finally:
            # Clean up: remove our custom handler
            root_logger.removeHandler(handler)
    
    def return_to_menu_after_simulation(self) -> None:
        """Return to the main menu after simulation completion."""
        try:
            # Get the body widget and switch back to menu
            body = self.query_one("Body")
            body.remove_children()
            body.mount(Menu())
        except Exception as e:
            # If there's an error returning to menu, just log it
            logging.error(f"Error returning to menu: {str(e)}")
    
    def return_to_menu_after_setup(self) -> None:
        """Return to the main menu after setup completion."""
        try:
            # Get the body widget and switch back to menu
            body = self.query_one("Body")
            body.remove_children()
            body.mount(Menu())
        except Exception as e:
            # If there's an error returning to menu, just log it
            logging.error(f"Error returning to menu: {str(e)}")



def main():
    """Main entry point for the mythologizer command."""
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = MythologizerApp()
    app.run()

if __name__ == "__main__":
    main()
