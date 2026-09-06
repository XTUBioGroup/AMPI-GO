

# # Match PDB IDs


# import os
# import glob
# import pandas as pd

# # ===================== Configuration =====================

# # 1) Input: Paths to two directories containing PPI files
# folder1 = r"ppi_from_species_2hop_train_100_10"
# folder2 = r"ppi_from_species_2hop_valid_100_10"

# # 2) Input: Dataset ID file (used to exclude existing sequences)
# #    Format: At least three columns; the third is a STRING ID and may be NA
# dataset_id_file = r"pdb_uniprot_string.txt"

# # 3) Input: Complete STRING sequence file (FASTA format)
# fasta_file = r"F:\download\protein.sequences.v12.0.fa\protein.sequences.v12.0.fa"

# # 4) Output: Path for the supplemental sequence file
# output_txt = r"extra_protein_sequences_combined.txt"

# # =================================================


# def collect_ppi_proteins_from_folders(folders):
#     """
#     Iterate over the specified directories and read protein1 and protein2 from all .txt files.
#     """
#     proteins = set()
#     total_files = 0

#     for folder in folders:
#         # Build the search path: folder/*.txt
#         search_pattern = os.path.join(folder, "*.txt")
#         files = glob.glob(search_pattern)
        
#         print(f"\n📂 Scanning directory: {folder}")
#         print(f"   Found {len(files)} TXT files")
        
#         total_files += len(files)

#         # Read files one by one
#         for i, path in enumerate(files):
#             # Print simple progress every 100 files
#             if (i + 1) % 500 == 0:
#                 print(f"   Processing file {i + 1}...")
            
#             try:
#                 # Read only the first two columns to save memory
#                 # STRING format is usually tab-delimited
#                 df = pd.read_csv(path, sep="\t", usecols=["protein1", "protein2"])
                
#                 # Convert both columns to sets and update
#                 proteins.update(df["protein1"].astype(str))
#                 proteins.update(df["protein2"].astype(str))
                
#             except ValueError:
#                 # The file may be empty or have incorrect column names
#                 pass
#             except Exception as e:
#                 print(f"   ❌ Failed to read {path}: {e}")

#     print(f"\n📊 Statistics:")
#     print(f"   Total files processed: {total_files}")
#     print(f"   Unique proteins in PPI data: {len(proteins)}")
#     return proteins


# def load_dataset_string_ids(filepath):
#     """
#     Read the third column of pdb_uniprot_string.txt.
#     """
#     ids = set()
#     if not os.path.exists(filepath):
#         print(f"❌ Dataset file not found: {filepath}")
#         return ids

#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
            
#             parts = line.split()  # Split on whitespace by default
#             if len(parts) < 3:
#                 continue
            
#             string_id = parts[2]
#             if string_id != "NA":
#                 ids.add(string_id)

#     print(f"📚 STRING IDs in the original dataset: {len(ids)}")
#     return ids


# def extract_sequences_from_fasta(fasta_path, wanted_ids):
#     """
#     Stream the FASTA file and extract only required IDs.
#     """
#     seqs = {}
#     current_id = None
#     current_seq_lines = []

#     print(f"\n🧬 Reading FASTA library: {fasta_path}")
#     print("   This may take several minutes; please wait...")

#     def flush():
#         nonlocal current_id, current_seq_lines
#         if current_id is not None and current_id in wanted_ids:
#             seqs[current_id] = "".join(current_seq_lines)
#         current_id = None
#         current_seq_lines = []

#     try:
#         with open(fasta_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.rstrip("\n")
#                 if not line:
#                     continue

#                 if line.startswith(">"):
#                     # Save the previous record
#                     flush()
#                     # Parse a new ID: >9606.ENSP00000123456 description...
#                     header = line[1:]
#                     # Use the part before the first space as the ID
#                     header_id = header.split()[0]
#                     current_id = header_id
#                 else:
#                     current_seq_lines.append(line)
            
#             # Process the final record
#             flush()

#     except FileNotFoundError:
#         print(f"❌ FASTA file not found: {fasta_path}")
#         return {}

#     print(f"✅ Successfully extracted sequences: {len(seqs)}")
#     missing = len(wanted_ids) - len(seqs)
#     if missing > 0:
#         print(f"⚠️ Warning: {missing} proteins were not found in FASTA (possibly obsolete IDs or missing species data)")
    
#     return seqs


# def main():
#     # 1. Collect all PPI proteins from both directories
#     #    Place folder1 and folder2 in a list
#     target_folders = [folder1, folder2]
#     ppi_ids = collect_ppi_proteins_from_folders(target_folders)

#     if not ppi_ids:
#         print("❌ No PPI proteins found; exiting.")
#         return

#     # 2. Read IDs from the existing dataset
#     dataset_ids = load_dataset_string_ids(dataset_id_file)

#     # 3. Compute the difference: required IDs = IDs in PPI - IDs already in the dataset
#     extra_ids = ppi_ids - dataset_ids
#     print(f"\n🔍 Proteins requiring supplemental sequences (difference): {len(extra_ids)}")

#     if not extra_ids:
#         print("✅ All PPI proteins already exist in the dataset; no supplementation required.")
#         return

#     # 4. Extract sequences from FASTA
#     seq_dict = extract_sequences_from_fasta(fasta_file, extra_ids)

#     # 5. Save results
#     print(f"\n💾 Saving to: {output_txt}")
#     with open(output_txt, "w", encoding="utf-8") as out:
#         # Write the header (optional)
#         # out.write("string_id\tsequence\n") 
#         for pid, seq in seq_dict.items():
#             out.write(f"{pid}\t{seq}\n")

#     print("🎉 All done!")


# if __name__ == "__main__":
#     main()



import os
import glob
import pandas as pd

# ===================== Configuration =====================

# 1) Input: PPI directory path (one directory only)
ppi_folder = r"ppi_from_species_2hop_valid_100_10_supplement"

# 2) Input: Existing sequence file (used for exclusion)
#    Format: First column is the STRING ID (tab-delimited)
existing_seq_file = r"extra_protein_sequences_combined.txt"

# 3) Input: Dataset ID file (also exclude IDs already present in the dataset)
#    (If extra_protein_sequences_combined.txt already contains all supplemental entries outside the dataset,
#     this step is an additional safeguard and should be retained.)
dataset_id_file = r"pdb_uniprot_string_supplement.txt"

# 4) Input: Complete STRING sequence library
fasta_file = r"F:\download\protein.sequences.v12.0.fa\protein.sequences.v12.0.fa"

# 5) Output: New incremental supplemental file
output_txt = r"extra_protein_sequences_combined_supplement.txt"

# =================================================


def collect_ppi_proteins_from_folder(folder):
    """
    Scan all .txt files in one directory and extract protein1 and protein2.
    """
    proteins = set()
    search_pattern = os.path.join(folder, "*.txt")
    files = glob.glob(search_pattern)
    
    print(f"\n📂 Scanning directory: {folder}")
    print(f"   Found {len(files)} TXT files")

    for i, path in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i + 1} files...", end="\r")
        
        try:
            # Read only the first two columns
            df = pd.read_csv(path, sep="\t", usecols=["protein1", "protein2"])
            proteins.update(df["protein1"].astype(str))
            proteins.update(df["protein2"].astype(str))
        except Exception:
            pass # Ignore empty or invalid files

    print(f"\n✅ Total unique proteins involved in directory PPI data: {len(proteins)}")
    return proteins


def load_existing_ids(filepath):
    """
    Read the first column of the existing sequence file (extra_protein_sequences_combined.txt).
    """
    ids = set()
    if not os.path.exists(filepath):
        print(f"⚠️ File does not exist; no exclusions will be applied: {filepath}")
        return ids

    print(f"📖 Loading the existing sequence list: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Assumed format: ID \t Sequence
            parts = line.split("\t")
            ids.add(parts[0])
    
    print(f"   -> Excluded {len(ids)} known sequences")
    return ids


def load_dataset_string_ids(filepath):
    """
    Read the third column of pdb_uniprot_string.txt.
    """
    ids = set()
    if not os.path.exists(filepath):
        return ids

    print(f"📖 Loading original dataset IDs: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) >= 3:
                string_id = parts[2]
                if string_id != "NA":
                    ids.add(string_id)

    print(f"   -> Excluded {len(ids)} original dataset IDs")
    return ids


def extract_sequences_from_fasta(fasta_path, wanted_ids):
    """
    Stream the FASTA file and extract only wanted_ids.
    """
    seqs = {}
    current_id = None
    current_seq_lines = []
    
    # Convert to a set for faster lookup
    wanted_set = set(wanted_ids)

    print(f"\n🧬 Scanning the FASTA library for {len(wanted_set)} sequences...")
    
    def flush():
        nonlocal current_id, current_seq_lines
        if current_id and current_id in wanted_set:
            seqs[current_id] = "".join(current_seq_lines)
        current_id = None
        current_seq_lines = []

    try:
        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue

                if line.startswith(">"):
                    flush()
                    # Parse ID: >9606.ENSP... description
                    header = line[1:]
                    current_id = header.split()[0]
                else:
                    current_seq_lines.append(line)
            flush() # Final record

    except FileNotFoundError:
        print(f"❌ Error: FASTA file not found: {fasta_path}")
        return {}

    print(f"✅ Successfully extracted: {len(seqs)}")
    return seqs


def main():
    # 1. Get all IDs from PPI data
    ppi_ids = collect_ppi_proteins_from_folder(ppi_folder)
    if not ppi_ids:
        return

    # 2. Get IDs to exclude (combined file + original dataset)
    existing_combined_ids = load_existing_ids(existing_seq_file)
    original_dataset_ids = load_dataset_string_ids(dataset_id_file)
    
    # Merge all known IDs
    all_known_ids = existing_combined_ids.union(original_dataset_ids)

    # 3. Compute the actual increment (difference)
    ids_to_fetch = ppi_ids - all_known_ids
    
    print(f"\n📊 Summary:")
    print(f"   Total PPI IDs:             {len(ppi_ids)}")
    print(f"   Existing-file IDs:         {len(existing_combined_ids)}")
    print(f"   Original-dataset IDs:      {len(original_dataset_ids)}")
    print(f"   ---------------------------")
    print(f"   🔥 IDs actually requiring supplementation: {len(ids_to_fetch)}")

    if not ids_to_fetch:
        print("\n🎉 No new IDs require supplementation; all sequences already exist!")
        return

    # 4. Extract sequences
    new_seqs = extract_sequences_from_fasta(fasta_file, ids_to_fetch)

    # 5. Save the supplemental file
    if new_seqs:
        print(f"\n💾 Saving incremental file: {output_txt}")
        with open(output_txt, "w", encoding="utf-8") as f:
            for pid, seq in new_seqs.items():
                f.write(f"{pid}\t{seq}\n")
        print("🎉 Complete!")
    else:
        print("⚠️ No sequences for the target IDs were found in FASTA.")

if __name__ == "__main__":
    main()
