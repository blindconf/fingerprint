import os

def remove_k_chars_from_filenames(directory: str, k: int, remove_from: str = "end") -> None:

    for filename in os.listdir(directory):

        full_path = os.path.join(directory, filename)
        
        if os.path.isfile(full_path) and full_path.endswith(".wav"):

            base, ext = os.path.splitext(filename)
            
            if len(base) <= k:
                print(f"Skipping '{filename}' because its base name is too short to remove {k} characters.")
                continue
            
            if remove_from == "start":
                new_base = base[k:]
            elif remove_from == "end":
                new_base = base[:-k]
            else:
                print(f"Invalid option for remove_from: {remove_from}. Use 'start' or 'end'.")
                continue
            
            new_filename = new_base + ext
            new_full_path = os.path.join(directory, new_filename)
            
            os.rename(full_path, new_full_path)
            print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":

    directory_path = ""
    k = 3
    
    remove_k_chars_from_filenames(directory_path, k, remove_from="end")
