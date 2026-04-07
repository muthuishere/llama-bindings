package com.example.llama.knowledge;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Persistent knowledge store backed by SQLite (via sqlite-jdbc).
 *
 * <p>Supports hybrid RRF (Reciprocal Rank Fusion) retrieval that combines
 * vector cosine-similarity search with SQLite FTS5 keyword search.
 *
 * <p>Usage:
 * <pre>{@code
 * try (KnowledgeStore store = new KnowledgeStore(":memory:")) {
 *     store.add("The sky is blue", new float[]{1.0f, 0.0f});
 *     List<Document> docs = store.search(new float[]{0.9f, 0.1f}, "sky", 5);
 * }
 * }</pre>
 */
public final class KnowledgeStore implements AutoCloseable {

    /** A retrieved knowledge document. */
    public static final class Document {
        public final long   id;
        public final String text;
        public final double score;

        Document(long id, String text, double score) {
            this.id    = id;
            this.text  = text;
            this.score = score;
        }
    }

    private final Connection conn;

    /**
     * Open (or create) a SQLite knowledge store at {@code dsn}.
     * Use {@code ":memory:"} for an ephemeral in-memory store.
     *
     * @param dsn JDBC SQLite DSN, e.g. {@code "jdbc:sqlite:/path/to/db"} or
     *            the bare path / {@code ":memory:"}
     * @throws SQLException if the database cannot be opened
     */
    public KnowledgeStore(String dsn) throws SQLException {
        String url = dsn.startsWith("jdbc:") ? dsn : "jdbc:sqlite:" + dsn;
        conn = DriverManager.getConnection(url);
        conn.setAutoCommit(true);
        migrate();
    }

    /** Initialise schema. */
    private void migrate() throws SQLException {
        try (Statement st = conn.createStatement()) {
            st.executeUpdate(
                "CREATE TABLE IF NOT EXISTS documents (" +
                "  id        INTEGER PRIMARY KEY AUTOINCREMENT," +
                "  text      TEXT    NOT NULL," +
                "  embedding BLOB    NOT NULL" +
                ")");
            st.executeUpdate(
                "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts" +
                " USING fts5(text, content=documents, content_rowid=id)");
            st.executeUpdate(
                "CREATE TRIGGER IF NOT EXISTS documents_ai" +
                " AFTER INSERT ON documents BEGIN" +
                "  INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);" +
                " END");
            st.executeUpdate(
                "CREATE TRIGGER IF NOT EXISTS documents_ad" +
                " AFTER DELETE ON documents BEGIN" +
                "  INSERT INTO documents_fts(documents_fts, rowid, text)" +
                "  VALUES ('delete', old.id, old.text);" +
                " END");
        }
    }

    /**
     * Add a document with its pre-computed embedding vector.
     *
     * @param text      document text (must not be empty)
     * @param embedding embedding vector (must not be empty)
     * @throws IllegalArgumentException if text or embedding is empty
     * @throws SQLException             on database error
     */
    public synchronized void add(String text, float[] embedding)
            throws IllegalArgumentException, SQLException {
        if (text == null || text.isEmpty()) {
            throw new IllegalArgumentException("KnowledgeStore: text must not be empty");
        }
        if (embedding == null || embedding.length == 0) {
            throw new IllegalArgumentException("KnowledgeStore: embedding must not be empty");
        }
        try (PreparedStatement ps = conn.prepareStatement(
                "INSERT INTO documents(text, embedding) VALUES (?, ?)")) {
            ps.setString(1, text);
            ps.setBytes(2, floatsToBytes(embedding));
            ps.executeUpdate();
        }
    }

    /**
     * Hybrid RRF search: combines vector cosine-similarity with FTS keyword search.
     *
     * @param queryVec   query embedding vector (must not be empty)
     * @param queryText  query text for keyword search (may be {@code null} / empty)
     * @param limit      maximum number of results
     * @return list of documents ordered by descending RRF score
     * @throws SQLException on database error
     */
    public synchronized List<Document> search(float[] queryVec, String queryText, int limit)
            throws SQLException {
        if (queryVec == null || queryVec.length == 0) {
            throw new IllegalArgumentException("KnowledgeStore: queryVec must not be empty");
        }
        if (limit <= 0) limit = 5;

        // 1. Vector similarity: full-table scan.
        record VecResult(long id, String text, double score) {}
        List<VecResult> vecResults = new ArrayList<>();

        try (Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(
                     "SELECT id, text, embedding FROM documents")) {
            while (rs.next()) {
                long   id        = rs.getLong(1);
                String text      = rs.getString(2);
                float[] stored   = bytesToFloats(rs.getBytes(3));
                double score     = cosineSimilarity(queryVec, stored);
                vecResults.add(new VecResult(id, text, score));
            }
        }

        vecResults.sort((a, b) -> Double.compare(b.score(), a.score()));
        Map<Long, Integer> vecRank = new HashMap<>();
        for (int i = 0; i < vecResults.size(); i++) {
            vecRank.put(vecResults.get(i).id(), i + 1);
        }

        // 2. FTS keyword search.
        Map<Long, Integer> ftsRank = new HashMap<>();
        if (queryText != null && !queryText.isEmpty()) {
            try (PreparedStatement ps = conn.prepareStatement(
                    "SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? " +
                    "ORDER BY rank LIMIT ?")) {
                ps.setString(1, sanitizeFtsQuery(queryText));
                ps.setInt(2, limit * 3);
                try (ResultSet rs = ps.executeQuery()) {
                    int rank = 1;
                    while (rs.next()) {
                        ftsRank.put(rs.getLong(1), rank++);
                    }
                }
            } catch (SQLException ignored) {
                // FTS errors are non-fatal; fall back to vector-only.
            }
        }

        // 3. RRF fusion.
        final int K = 60;
        List<Document> entries = new ArrayList<>(vecResults.size());
        for (VecResult vr : vecResults) {
            int vecR = vecRank.getOrDefault(vr.id(), Integer.MAX_VALUE);
            int ftsR = ftsRank.getOrDefault(vr.id(), Integer.MAX_VALUE);
            double rrf =
                (vecR < Integer.MAX_VALUE ? 1.0 / (K + vecR) : 0.0) +
                (ftsR < Integer.MAX_VALUE ? 1.0 / (K + ftsR) : 0.0);
            entries.add(new Document(vr.id(), vr.text(), rrf));
        }

        entries.sort((a, b) -> Double.compare(b.score, a.score));
        if (entries.size() > limit) {
            entries = entries.subList(0, limit);
        }
        return new ArrayList<>(entries);
    }

    /** Close the underlying SQLite connection. */
    @Override
    public void close() {
        try { conn.close(); } catch (SQLException ignored) {}
    }

    // ──────────────────────────────────────────────────────────────────────
    // Math helpers
    // ──────────────────────────────────────────────────────────────────────

    private static double cosineSimilarity(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    // ──────────────────────────────────────────────────────────────────────
    // Serialisation helpers
    // ──────────────────────────────────────────────────────────────────────

    private static byte[] floatsToBytes(float[] v) {
        ByteBuffer buf = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : v) buf.putFloat(f);
        return buf.array();
    }

    private static float[] bytesToFloats(byte[] b) {
        if (b == null) return new float[0];
        ByteBuffer buf = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        float[] v = new float[b.length / 4];
        for (int i = 0; i < v.length; i++) v[i] = buf.getFloat();
        return v;
    }

    /** Escape FTS query to avoid syntax errors with special characters. */
    private static String sanitizeFtsQuery(String query) {
        // Remove FTS5 special characters and wrap in double-quotes.
        return "\"" + query.replace("\"", " ") + "\"";
    }
}
