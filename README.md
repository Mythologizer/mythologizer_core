# Mythologizer

A simulation framework for modeling cultural evolution through myth exchange between agents.

## Installation

### Prerequisites

- Python 3.8 or higher
- TODO
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mytholgoizer_core
   ```

2. **Install dependencies using uv**
   ```bash
   # Install uv if you don't have it
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv sync
   ```

3. **Set up PostgreSQL database**
   TODO

4. **Configure environment**
   - Copy `.env.example` to `.env` (if it exists)
   - Update the `.env` file with your PostgreSQL credentials:
     ```
     POSTGRES_HOST=localhost
     POSTGRES_PORT=5432
     POSTGRES_USER=your_username
     POSTGRES_PASSWORD=your_password
     POSTGRES_DB=your_database_name
     ```

## Running the Application

### GUI Mode (Recommended)

1. **Start the GUI application**
   ```bash
   uv run python -m mythologizer_core.ui.app
   ```

2. **Configure settings in the GUI**
   - Click "Settings" to configure database connection and simulation parameters
   - Save your settings

3. **Setup simulation**
   - Click "Setup Simulation" to initialize the database with agents and myths
   - This will create the necessary tables and populate them with initial data

4. **Run simulation**
   - Click "Run Simulation" to start the myth exchange simulation
   - Monitor the progress in the log view

### Command Line Mode

1. **Setup simulation**
   ```bash
   uv run python -m mythologizer_core.setup
   ```

2. **Run simulation**
   ```bash
   uv run python -m mythologizer_core.simulation
   ```

## Settings Guide

### Database Settings

- **Postgres Host**: Database server hostname (default: localhost)
- **Postgres Port**: Database server port (default: 5432)
- **Postgres User**: Database username
- **Postgres Password**: Database password
- **Postgres Database**: Database name

### Simulation Settings

- **Sentence Embedding Model**: Model for text embeddings (default: all-MiniLM-L6-v2)
- **Agent Attributes File**: Python file defining agent attributes (default: agent_attributes.py)
- **Mythemes File**: Text file containing mythemes (default: mythemes.txt)
- **Initial Cultures File**: Optional file for initial cultures

### Epoch Settings

- **Number of Epochs**: Number of simulation epochs to run (optional)
- **Number of Interactions per Epoch**: Number of myth exchanges per epoch (default: 4)
- **Max Number of Listeners per Interaction**: Maximum listeners per myth exchange (default: 3)
- **Minimum Epoch Time**: Minimum time in seconds each epoch must run (default: 10)

### Myth Exchange Parameters

#### Event and Culture Weights
- **Event Weight**: Weight for event influence on myth selection (0.0-1.0, default: 0.0)
- **Culture Weight**: Weight for cultural influence on myths (0.0-1.0, default: 0.0)
- **Weight of Attribute Embeddings**: Weight for agent attribute embeddings (0.0-1.0, default: 1.0)

#### Thresholds
- **New Myth Threshold**: Threshold for creating new myths vs updating existing ones (0.0-1.0, default: 0.5)
- **Max Threshold for Listener Myth**: Maximum threshold for listener myth selection (0.0-1.0, default: 0.5)

#### Retention Factors
- **Retention Remember Factor**: Factor to increase speaker's myth retention (0.0-1.0, default: 0.1)
- **Retention Forget Factor**: Factor to decrease retention of other speaker myths (0.0-1.0, default: 0.05)

#### Mutation Probabilities
- **Deletion Probability**: Probability of deleting a mytheme during mutation (0.0-1.0, default: 0.2)
- **Mutation Probability**: Probability of mutating mythemes (0.0-1.0, default: 0.7)
- **Reordering Probability**: Probability of reordering mythemes (0.0-1.0, default: 0.3)

### Parameter Effects

#### Epoch Settings
- **Number of Epochs**: Controls total simulation duration (leave empty for infinite)
- **Number of Interactions per Epoch**: Higher values create more myth exchanges per epoch, leading to faster cultural evolution
- **Max Number of Listeners per Interaction**: Higher values allow myths to spread to more agents simultaneously, increasing myth propagation speed
- **Minimum Epoch Time**: Ensures each epoch runs for at least this many seconds, useful for real-time monitoring or rate limiting

#### Event and Culture Weights
- Higher event weights make myths more influenced by current events
- Higher culture weights make myths more influenced by cultural background
- Higher attribute embedding weights make myths more influenced by agent characteristics

#### Thresholds
- Lower new myth thresholds create more new myths
- Higher max threshold for listener myths makes myth selection more selective

#### Retention Factors
- Higher remember factors make agents remember told myths better
- Higher forget factors make agents forget other myths faster

#### Mutation Probabilities
- Higher deletion probabilities result in shorter myths over time
- Higher mutation probabilities create more diverse myth variations
- Higher reordering probabilities change myth structure more frequently

## File Structure

- `mythologizer_core/`: Main application code
  - `ui/`: GUI application
  - `epoch/`: Epoch processing logic
  - `myth_exchange/`: Myth exchange algorithms
  - `agent_attribute/`: Agent attribute definitions
  - `types/`: Type definitions
  - `utils/`: Utility functions
- `agent_attributes.py`: Agent attribute definitions
- `mythemes.txt`: Mytheme definitions
- `simulation_config.json`: Simulation configuration (auto-generated)

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check database credentials in `.env` file
- Ensure database exists and is accessible

### Missing Dependencies
- Run `uv sync` to install all dependencies
- Check Python version (requires 3.8+)

### Simulation Errors
- Check log output for specific error messages
- Verify all required files exist (agent_attributes.py, mythemes.txt)
- Ensure database tables are properly initialized

## License

[Add your license information here]
