import fire  # <-- Import fire
import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure
import sys

def test_mongo(connection_string: str):
    """
    Tests a MongoDB connection string by attempting to ping the server.
    
    :param connection_string: The full MongoDB connection URI.
    """
    
    print(f"Attempting to connect to: {connection_string[:50]}...")
    client = None
    
    try:
        # 1. Create a client from the command-line argument
        # We set a 5-second timeout to fail fast
        client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        
        # 2. Force a connection test by "pinging" the admin database
        client.admin.command('ping')
        
        print("\n✅ Connection successful!")

    except OperationFailure as e:
        # This happens if the username/password is wrong
        print("\n❌ Authentication failed:")
        print(e.details)
        sys.exit(1) # Exit with an error code
        
    except ConnectionFailure as e:
        # This happens if the server is unreachable (check Network Access in Atlas)
        print("\n❌ Connection failed (Network error or IP not whitelisted?):")
        print(e)
        sys.exit(1)
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

    finally:
        # 3. Clean up the connection
        if client:
            client.close()

if __name__ == "__main__":
    fire.Fire(test_mongo) # <-- This makes your function a CLI
