from accelerate.utils import gather_object
from accelerate import Accelerator

def test():
    acc = Accelerator()
    local_data = [{"a": 1, "rank": acc.process_index}, {"a": 2, "rank": acc.process_index}]
    gathered = gather_object(local_data)
    if acc.is_main_process:
        print(f"Gathered type: {type(gathered)}")
        print(f"Gathered len: {len(gathered)}")
        print(f"Gathered content: {gathered}")
        
        # Test flattening vs direct use
        try:
            print("Attempting to access as list of dicts:")
            print(gathered[0]['a'])
            print("Success!")
        except Exception as e:
            print(f"Failed direct access: {e}")
            
        try:
            print("Attempting to flatten (assuming list of lists):")
            flattened = [item for sublist in gathered for item in sublist]
            print(f"Flattened content: {flattened}")
        except Exception as e:
            print(f"Failed flattening: {e}")

if __name__ == "__main__":
    test()
