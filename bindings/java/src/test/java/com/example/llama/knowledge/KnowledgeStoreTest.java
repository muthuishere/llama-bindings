package com.example.llama.knowledge;

import org.junit.jupiter.api.Test;

import java.sql.SQLException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class KnowledgeStoreTest {

    @Test
    void addAndSearchReturnsRelevantDocument() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            store.add("the sky is blue",   new float[]{1.0f, 0.0f});
            store.add("the grass is green", new float[]{0.0f, 1.0f});

            List<KnowledgeStore.Document> docs = store.search(
                    new float[]{1.0f, 0.0f}, "sky", 5);

            assertFalse(docs.isEmpty());
            assertEquals("the sky is blue", docs.get(0).text);
        }
    }

    @Test
    void searchEmptyStoreReturnsEmpty() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            List<KnowledgeStore.Document> docs = store.search(
                    new float[]{1.0f, 0.0f}, "", 5);
            assertTrue(docs.isEmpty());
        }
    }

    @Test
    void addEmptyTextThrows() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            assertThrows(IllegalArgumentException.class,
                    () -> store.add("", new float[]{1.0f}));
        }
    }

    @Test
    void addEmptyEmbeddingThrows() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            assertThrows(IllegalArgumentException.class,
                    () -> store.add("text", new float[]{}));
        }
    }

    @Test
    void searchEmptyVectorThrows() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            assertThrows(IllegalArgumentException.class,
                    () -> store.search(new float[]{}, "", 5));
        }
    }

    @Test
    void limitIsRespected() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            for (int i = 0; i < 10; i++) {
                store.add("doc " + i, new float[]{(float) i / 10, 1.0f - (float) i / 10});
            }
            List<KnowledgeStore.Document> docs = store.search(
                    new float[]{1.0f, 0.0f}, "", 3);
            assertTrue(docs.size() <= 3);
        }
    }

    @Test
    void vectorRankingIsCorrect() throws SQLException {
        try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
            store.add("A", new float[]{1.0f, 0.0f, 0.0f});
            store.add("B", new float[]{0.0f, 1.0f, 0.0f});
            store.add("C", new float[]{0.0f, 0.0f, 1.0f});

            List<KnowledgeStore.Document> docs = store.search(
                    new float[]{0.9f, 0.1f, 0.0f}, "", 3);
            assertFalse(docs.isEmpty());
            assertEquals("A", docs.get(0).text);
        }
    }
}
