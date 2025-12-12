import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np


#Memory Storage Layer
class MemoryStore:
    """Persistent storage layer for agent memories."""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with memory schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                feedback_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                feedback_type TEXT,
                feedback_content TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_memory(self, memory_type: str, content: str, 
                   embedding: Optional[List[float]] = None,
                   confidence: float = 1.0, 
                   metadata: Optional[Dict] = None) -> int:
        """Save a new memory entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        embedding_str = json.dumps(embedding) if embedding else None
        metadata_str = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO memories (type, content, embedding, timestamp, confidence, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_type, content, embedding_str, timestamp, confidence, metadata_str))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
    
    def get_memory(self, memory_id: int) -> Optional[Dict]:
        """Retrieve a specific memory by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def get_all_memories(self, memory_type: Optional[str] = None) -> List[Dict]:
        """Retrieve all memories, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute("SELECT * FROM memories WHERE type = ? ORDER BY timestamp DESC", 
                         (memory_type,))
        else:
            cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def update_memory(self, memory_id: int, updates: Dict):
        """Update a memory entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic UPDATE query
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [memory_id]
        
        cursor.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
        conn.commit()
        conn.close()
    
    def delete_memory(self, memory_id: int):
        """Delete a memory entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        cursor.execute("DELETE FROM feedback WHERE memory_id = ?", (memory_id,))
        
        conn.commit()
        conn.close()
    
    def save_feedback(self, memory_id: int, feedback_type: str, 
                     feedback_content: str):
        """Save feedback for a memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO feedback (memory_id, feedback_type, feedback_content, timestamp)
            VALUES (?, ?, ?, ?)
        """, (memory_id, feedback_type, feedback_content, timestamp))
        
        conn.commit()
        conn.close()
    
    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        return {
            'id': row[0],
            'type': row[1],
            'content': row[2],
            'embedding': json.loads(row[3]) if row[3] else None,
            'timestamp': row[4],
            'confidence': row[5],
            'feedback_count': row[6],
            'last_accessed': row[7],
            'metadata': json.loads(row[8]) if row[8] else None
        }

class MemoryRetriever:
    """Retrieves relevant memories using semantic similarity and recency."""
    
    def __init__(self, memory_store: MemoryStore, embedding_model=None):
        self.memory_store = memory_store
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str, top_k: int = 5, 
                memory_type: Optional[str] = None,
                recency_weight: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant memories based on semantic similarity and recency.
        
        Args:
            query: The query string to match against
            top_k: Number of memories to retrieve
            memory_type: Filter by memory type
            recency_weight: Weight for recency scoring (0-1)
        """
        memories = self.memory_store.get_all_memories(memory_type)
        
        if not memories:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Score each memory
        scored_memories = []
        now = datetime.now()
        
        for memory in memories:
            # Calculate semantic similarity
            if query_embedding and memory['embedding']:
                similarity = self._cosine_similarity(query_embedding, memory['embedding'])
            else:
                similarity = 0.0
            
            # Calculate recency score
            memory_time = datetime.fromisoformat(memory['timestamp'])
            time_diff = (now - memory_time).total_seconds()
            recency_score = 1.0 / (1.0 + time_diff / 86400)  # Decay over days
            
            # Combined score
            final_score = (
                (1 - recency_weight) * similarity + 
                recency_weight * recency_score
            ) * memory['confidence']
            
            scored_memories.append({
                **memory,
                'relevance_score': final_score
            })
        
        # Sort by score and return top k
        scored_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update last_accessed for retrieved memories
        for memory in scored_memories[:top_k]:
            self.memory_store.update_memory(
                memory['id'],
                {'last_accessed': datetime.now().isoformat()}
            )
        
        return scored_memories[:top_k]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using the configured model."""
        if not self.embedding_model:
            return None
        
        # Placeholder - implement with your embedding model
        # Example: OpenAI, Sentence Transformers, Google embeddings, etc.
        try:
            # For Google Gemini embeddings:
            # response = self.embedding_model.embed_content(
            #     model="models/text-embedding-004",
            #     content=text
            # )
            # return response['embedding']
            
            # Dummy implementation for now
            return [0.0] * 768
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class FeedbackHandler:
    """Handles user feedback and updates memory accordingly."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory_store = memory_store
    
    def handle_feedback(self, memory_id: int, feedback_type: str, 
                       feedback_content: str = ""):
        """
        Process user feedback and update memory.
        
        Feedback types:
        - 'positive': Increases confidence
        - 'negative': Decreases confidence
        - 'corrective': Updates content based on correction
        - 'delete': Marks for deletion
        """
        memory = self.memory_store.get_memory(memory_id)
        
        if not memory:
            print(f"Memory {memory_id} not found")
            return
        
        # Save feedback record
        self.memory_store.save_feedback(memory_id, feedback_type, feedback_content)
        
        # Update memory based on feedback type
        if feedback_type == 'positive':
            self._handle_positive_feedback(memory)
        elif feedback_type == 'negative':
            self._handle_negative_feedback(memory)
        elif feedback_type == 'corrective':
            self._handle_corrective_feedback(memory, feedback_content)
        elif feedback_type == 'delete':
            self._handle_delete_feedback(memory)
    
    def _handle_positive_feedback(self, memory: Dict):
        """Increase confidence score."""
        new_confidence = min(1.0, memory['confidence'] + 0.1)
        self.memory_store.update_memory(
            memory['id'],
            {
                'confidence': new_confidence,
                'feedback_count': memory['feedback_count'] + 1
            }
        )
    
    def _handle_negative_feedback(self, memory: Dict):
        """Decrease confidence score, delete if too low."""
        new_confidence = max(0.0, memory['confidence'] - 0.2)
        
        if new_confidence <= 0.1:
            self.memory_store.delete_memory(memory['id'])
        else:
            self.memory_store.update_memory(
                memory['id'],
                {
                    'confidence': new_confidence,
                    'feedback_count': memory['feedback_count'] + 1
                }
            )
    
    def _handle_corrective_feedback(self, memory: Dict, correction: str):
        """Update memory content with correction."""
        self.memory_store.update_memory(
            memory['id'],
            {
                'content': correction,
                'confidence': 0.9,  # Slight reduction due to correction
                'feedback_count': memory['feedback_count'] + 1
            }
        )
    
    def _handle_delete_feedback(self, memory: Dict):
        """Mark memory for deletion."""
        self.memory_store.delete_memory(memory['id'])

class ContextBuilder:
    """Builds context from retrieved memories for agent prompts."""
    
    def __init__(self, retriever: MemoryRetriever):
        self.retriever = retriever
    
    def build_context(self, query: str, memory_type: Optional[str] = None,
                     max_memories: int = 5) -> str:
        """
        Build context string from relevant memories.
        
        Returns formatted string to inject into agent prompt.
        """
        memories = self.retriever.retrieve(
            query, 
            top_k=max_memories,
            memory_type=memory_type
        )
        
        if not memories:
            return ""
        
        context = "\n\nRelevant memories from past interactions:\n"
        
        for i, memory in enumerate(memories, 1):
            context += f"\n{i}. [{memory['type']}] {memory['content']}"
            if memory.get('metadata'):
                context += f" (Confidence: {memory['confidence']:.2f})"
        
        context += "\n\nUse these memories to provide more personalized and context-aware responses.\n"
        
        return context

class AgentMemoryIntegration:
    """Integrates memory system with agents."""
    
    def __init__(self, memory_store: MemoryStore, 
                 retriever: MemoryRetriever,
                 feedback_handler: FeedbackHandler,
                 context_builder: ContextBuilder):
        self.memory_store = memory_store
        self.retriever = retriever
        self.feedback_handler = feedback_handler
        self.context_builder = context_builder
    
    def pre_run_hook(self, user_query: str, memory_type: Optional[str] = None) -> str:
        """
        Called before agent runs - builds context from memories.
        
        Returns context string to inject into prompt.
        """
        return self.context_builder.build_context(user_query, memory_type)
    
    def post_run_hook(self, user_query: str, agent_response: str,
                     memory_type: str = "interaction",
                     metadata: Optional[Dict] = None):
        """
        Called after agent runs - logs interaction to memory.
        
        Returns memory_id for potential feedback.
        """
        content = f"User: {user_query}\nAgent: {agent_response}"
        
        memory_id = self.memory_store.save_memory(
            memory_type=memory_type,
            content=content,
            metadata=metadata or {}
        )
        
        return memory_id
    
    def apply_feedback(self, memory_id: int, feedback_type: str,
                      feedback_content: str = ""):
        """Apply user feedback to a memory."""
        self.feedback_handler.handle_feedback(
            memory_id,
            feedback_type,
            feedback_content
        )