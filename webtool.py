import web_tool_pipeline
import os
def process_image_and_text(image_path, text_path):
    save_directory = 'patient_notes'
    os.makedirs(save_directory, exist_ok=True)

    # Create a filename using a unique identifier, e.g., based on current timestamp or other method
    file_name = "patient_notes.txt"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w') as file:
        file.write(text_path)
    phenotype_list= web_tool_pipeline.run_main(image_path, file_path)
    if phenotype_list is None:
        phenotype_list=["No finding"]
    return phenotype_list