#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2023 RhapsodyAI
#
# @author: bokai xu <bokesyo2000@gmail.com>
# @date: 2024/07/13
#


import tqdm
from PIL import Image
import hashlib
import torch
import fitz
import threading
import gradio as gr


def get_image_md5(img: Image.Image):
    img_byte_array = img.tobytes()
    hash_md5 = hashlib.md5()
    hash_md5.update(img_byte_array)
    hex_digest = hash_md5.hexdigest()
    return hex_digest

def pdf_to_images(pdf_path, dpi=100):
    doc = fitz.open(pdf_path)
    images = []
    for page in tqdm.tqdm(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def calculate_md5_from_binary(binary_data):
    hash_md5 = hashlib.md5()
    hash_md5.update(binary_data)
    return hash_md5.hexdigest()

class PDFVisualRetrieval:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.reps = {}
        self.images = {}
        self.lock = threading.Lock()
    
    def retrieve(self, knowledge_base: str, query: str, topk: int):
        doc_reps = list(self.reps[knowledge_base].values())
        query_with_instruction = "Represent this query for retrieving relavant document: " + query
        with torch.no_grad():
            query_rep = self.model(text=[query_with_instruction], image=[None], tokenizer=self.tokenizer).reps.squeeze(0)
        doc_reps_cat = torch.stack(doc_reps, dim=0)
        similarities = torch.matmul(query_rep, doc_reps_cat.T)
        topk_values, topk_doc_ids = torch.topk(similarities, k=topk)
        topk_values_np = topk_values.cpu().numpy()
        topk_doc_ids_np = topk_doc_ids.cpu().numpy()
        similarities_np = similarities.cpu().numpy()
        all_images_doc_list = list(self.images[knowledge_base].values())
        images_topk = [all_images_doc_list[idx] for idx in topk_doc_ids_np]
        return topk_doc_ids_np, topk_values_np, images_topk
    
    def add_pdf(self, knowledge_base_name: str, pdf_file_path: str, dpi: int = 100):
        if knowledge_base_name not in self.reps:
            self.reps[knowledge_base_name] = {}
        if knowledge_base_name not in self.images:
            self.images[knowledge_base_name] = {}
        doc = fitz.open(pdf_file_path)
        print("model encoding images..")
        for page in tqdm.tqdm(doc):
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_md5 = get_image_md5(image)
            with torch.no_grad():
                reps = self.model(text=[''], image=[image], tokenizer=self.tokenizer).reps
            self.reps[knowledge_base_name][image_md5] = reps.squeeze(0)
            self.images[knowledge_base_name][image_md5] = image
        return

    def add_pdf_gradio(self, pdf_file_binary, progress=gr.Progress()):
        knowledge_base_name = calculate_md5_from_binary(pdf_file_binary)
        if knowledge_base_name not in self.reps:
            self.reps[knowledge_base_name] = {}
        else:
            return knowledge_base_name
        if knowledge_base_name not in self.images:
            self.images[knowledge_base_name] = {}
        dpi = 100
        doc = fitz.open("pdf", pdf_file_binary)
        for page in progress.tqdm(doc):
            with self.lock: # because we hope one 16G gpu only process one image at the same time
                pix = page.get_pixmap(dpi=dpi)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image_md5 = get_image_md5(image)
                with torch.no_grad():
                    reps = self.model(text=[''], image=[image], tokenizer=self.tokenizer).reps
                self.reps[knowledge_base_name][image_md5] = reps.squeeze(0)
                self.images[knowledge_base_name][image_md5] = image
        return knowledge_base_name

    def retrieve_gradio(self, knowledge_base: str, query: str, topk: int):
        doc_reps = list(self.reps[knowledge_base].values())
        query_with_instruction = "Represent this query for retrieving relavant document: " + query
        with torch.no_grad():
            query_rep = self.model(text=[query_with_instruction], image=[None], tokenizer=self.tokenizer).reps.squeeze(0)
        doc_reps_cat = torch.stack(doc_reps, dim=0)
        similarities = torch.matmul(query_rep, doc_reps_cat.T)
        topk_values, topk_doc_ids = torch.topk(similarities, k=topk)
        topk_values_np = topk_values.cpu().numpy()
        topk_doc_ids_np = topk_doc_ids.cpu().numpy()
        similarities_np = similarities.cpu().numpy()
        all_images_doc_list = list(self.images[knowledge_base].values())
        images_topk = [all_images_doc_list[idx] for idx in topk_doc_ids_np]
        return images_topk


if __name__ == "__main__":
    from transformers import AutoModel
    from transformers import AutoTokenizer
    from PIL import Image
    import torch
    
    device = 'cuda:0'
    
    # Load model, be sure to substitute `model_path` by your model path 
    model_path = '/home/jeeves/xubokai/minicpm-visual-embedding-v0' # replace with your local model path
    # pdf_path = "/home/jeeves/xubokai/minicpm-visual-embedding-v0/2406.07422v1.pdf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    
    retriever = PDFVisualRetrieval(model=model, tokenizer=tokenizer)
    
    # topk_doc_ids_np, topk_values_np, images_topk = retriever.retrieve(knowledge_base='test', query='what is the number of VQ of this kind of codec method?', topk=1)
    # # 2
    # topk_doc_ids_np, topk_values_np, images_topk = retriever.retrieve(knowledge_base='test', query='the training loss curve of this paper?', topk=1)
    # # 3
    # topk_doc_ids_np, topk_values_np, images_topk = retriever.retrieve(knowledge_base='test', query='the experiment table?', topk=1)
    # # 2
    
    with gr.Blocks() as app:
        gr.Markdown("# Memex: OCR-free Visual Document Retrieval @RhapsodyAI")
        
        with gr.Row():
            file_input = gr.File(type="binary", label="Upload PDF")
            file_result = gr.Text(label="Knowledge Base ID (remember this!)")
            process_button = gr.Button("Process PDF")
        
        process_button.click(retriever.add_pdf_gradio, inputs=[file_input], outputs=file_result)

        with gr.Row():
            kb_id_input = gr.Text(label="Your Knowledge Base ID")
            query_input = gr.Text(label="Your Queston")
            topk_input = inputs=gr.Number(value=1, minimum=1, maximum=5, step=1, label="Top K")
            retrieve_button = gr.Button("Retrieve")
        
        with gr.Row():
            images_output = gr.Gallery(label="Retrieved Pages")
        
        retrieve_button.click(retriever.retrieve_gradio, inputs=[kb_id_input, query_input, topk_input], outputs=images_output)

    app.launch()

