from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

from textual.app import App, ComposeResult
from textual.widgets import Static, Log, Button
from textual.containers import Horizontal, HorizontalGroup
from textual import work
from textual.reactive import reactive
import asyncio
from art import *
from mythologizer_postgres.db import ping_db_basic
from mythologizer_postgres.connectors import get_simulation_status
from typing import Optional
from datetime import datetime


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
                connection_status = ping_db_basic()
                self.is_connected = connection_status is not False
            except Exception:
                self.is_connected = False
            
            await self.update_status_display()
            
            # If connected, update simulation status
            if self.is_connected:
                await self.update_simulation_status()
            
            await asyncio.sleep(1)
    
    async def update_status_display(self) -> None:
        """Update the status display based on connection state."""
        status_widget = self.query_one("#db_status")
        
        if self.is_connected:
            status_widget.update("Connected")
        else:
            status_widget.update("Disconnected")
    
    async def update_simulation_status(self) -> None:
        """Update simulation status when database is connected."""
        try:
            # Get simulation status from the database
            status = get_simulation_status()
            
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
        except Exception:
            # If there's an error, keep the last known values
            pass

class Body(Static):
    """Body widget."""
    
    def compose(self) -> ComposeResult:
        yield Menu()

class Menu(Static):
    """Menu widget."""
    
    def compose(self) -> ComposeResult:
        yield Button("Settings", classes="menu-item")
        yield Button("Setup Simulation", classes="menu-item")
        yield Button("Run Simulation", classes="menu-item")

class Footer(Static):
    """Footer widget."""
    def compose(self) -> ComposeResult:
        text = "Visit [bold]https://mythologizer.org[/bold] to learn more aswell as to create your own agents and browse though past myths."
        yield Static(text, classes="footer-text", markup=True)

    

class MythologizerApp(App):
    """Simple Hello World Textual app."""
    CSS_PATH = "app.tcss"
    
    
    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()
        yield SecondaryHeader()
        yield Body()
        yield Footer()



def main():
    """Main entry point for the mythologizer command."""

    app = MythologizerApp()
    app.run()

if __name__ == "__main__":
    main()
