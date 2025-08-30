import dotenv
import traceback
import numpy as np

dotenv.load_dotenv()

# Import the enhanced function with debugging
from mythologizer_postgres.connectors.mythicalgebra import get_myth_matrices_and_embedding_ids

def test_simulation_environment_loading():
    """Test that the simulation function can load environment variables properly."""
    print("=== Testing Simulation Environment Loading ===")
    
    try:
        # Import the simulation function
        from mythologizer_core.simulation import _pre_simulation_checks
        
        # Test the pre-simulation checks (this will load environment variables)
        print("Testing _pre_simulation_checks function...")
        _pre_simulation_checks("all-MiniLM-L6-v2")
        print("✓ _pre_simulation_checks completed successfully")
        
        # Test myth retrieval after environment loading
        print("\nTesting myth retrieval after environment loading...")
        myth_matrix, mytheme_ids = get_myth_matrices_and_embedding_ids(13)
        print(f"✓ Myth retrieval successful - Matrix shape: {myth_matrix.shape}")
        
    except Exception as e:
        print(f"✗ Environment loading test failed: {str(e)}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()

def test_with_enhanced_debugging():
    """Test with the enhanced debugging function to catch the exact source of string '[' errors."""
    print("\n=== Enhanced Debugging Test ===")
    print("This will use the enhanced debugging function to catch string '[' errors.")
    
    # Test myth IDs that might have issues
    test_myth_ids = [13, 14, 15, 16, 17, 18, 19, 20]
    
    for myth_id in test_myth_ids:
        print(f"\n--- Testing Myth ID {myth_id} ---")
        try:
            # This will use the enhanced debugging function
            myth_matrix, mytheme_ids = get_myth_matrices_and_embedding_ids(myth_id)
            print(f"✓ Myth ID {myth_id} - SUCCESS")
            print(f"  Matrix shape: {myth_matrix.shape}")
            print(f"  Mytheme IDs: {mytheme_ids}")
            
        except Exception as e:
            print(f"✗ Myth ID {myth_id} - FAILED")
            print(f"  Error: {str(e)}")
            if "could not convert string to float" in str(e):
                print(f"  *** THIS IS THE EXACT ERROR YOU'RE SEEING! ***")
            print(f"  Error type: {type(e)}")
            traceback.print_exc()
            break  # Stop on first error to focus on the problematic myth

if __name__ == "__main__":
    # Test environment loading in simulation
    test_simulation_environment_loading()
    
    # Run the enhanced debugging test
    test_with_enhanced_debugging()
    
    print("\n=== Summary ===")
    print("If both tests pass, the environment loading issue has been resolved.")
    print("The simulation should now work properly when run from Textual.")