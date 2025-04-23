import web_tool_pipeline

def process_image_and_text(image_path, text_path):
    phenotype_list= web_tool_pipeline.run_main(image_path, text_path)
    return phenotype_list