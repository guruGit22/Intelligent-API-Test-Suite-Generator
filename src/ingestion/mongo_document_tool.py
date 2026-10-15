from __future__ import annotations
from datetime import datetime,timezone
from pymongo import MongoClient
from config import settings
class MongoDocumentRepository:
    def __init__(self,uri=None,database=None,collection=None,chunk_collection=None):
        if not (settings.mongo_uri or uri): raise RuntimeError("MONGO_URI is not configured")
        self.client=MongoClient(uri or settings.mongo_uri,serverSelectionTimeoutMS=5000)
        self.database=self.client[database or settings.mongo_database]
        self.collection=self.database[collection or settings.mongo_document_collection]
        self.chunk_collection=self.database[chunk_collection or settings.mongo_chunk_collection]
    def ping(self): self.client.admin.command("ping"); return True
    def save_document(self,document_id,filename,document_type,raw,normalized,classification):
        self.collection.replace_one({"document_id":document_id},{"document_id":document_id,"file_name":filename,"document_type":document_type,"classification":classification,"raw":raw,"normalized":normalized,"ingested_at":datetime.now(timezone.utc).isoformat()},upsert=True)
    def save_chunks(self,chunks):
        if chunks:
            self.chunk_collection.delete_many({"document_id":chunks[0]["document_id"]}); self.chunk_collection.insert_many(chunks)
    def get_document(self,document_id): return self.collection.find_one({"document_id":document_id},{"_id":0})
    def get_chunks(self,document_id): return list(self.chunk_collection.find({"document_id":document_id},{"_id":0}))
    def get_all_chunks(self): return list(self.chunk_collection.find({},{"_id":0}))
class DocumentUnderstandingTool:
    def __init__(self,repository): self.repository=repository
    def understand(self,document_id):
        doc=self.repository.get_document(document_id)
        if not doc: raise KeyError(document_id)
        return {"document_id":document_id,"document_type":doc.get("document_type"),"classification":doc.get("classification"),"common_parameters":doc.get("normalized"),"chunks":self.repository.get_chunks(document_id)}
