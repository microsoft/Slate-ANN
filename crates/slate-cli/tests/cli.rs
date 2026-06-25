//! End-to-end CLI test: build a bundle from a plain-text vector file, then
//! query it through the actual `slate` binary and check the ranking.

use std::fs;
use std::process::Command;

/// Path to the compiled `slate` binary for this test run.
fn slate_bin() -> &'static str {
    env!("CARGO_BIN_EXE_slate")
}

#[test]
fn build_then_query_returns_nearest_first() {
    let dir = tempfile::tempdir().unwrap();
    let vectors_path = dir.path().join("vectors.txt");
    let query_path = dir.path().join("query.txt");
    let bundle_path = dir.path().join("bundle");

    // Five 4-d vectors. Row index 2 is an exact match for the query below, so
    // it must come back as the nearest neighbour under L2.
    fs::write(
        &vectors_path,
        "0 0 0 0\n\
         9 9 9 9\n\
         1 2 3 4\n\
         5 5 5 5\n\
         -1 -2 -3 -4\n",
    )
    .unwrap();
    fs::write(&query_path, "1 2 3 4\n").unwrap();

    let build = Command::new(slate_bin())
        .arg("build")
        .arg(&vectors_path)
        .arg(&bundle_path)
        .status()
        .expect("spawn slate build");
    assert!(build.success(), "`slate build` failed");

    // The bundle directory and its documented files should exist.
    assert!(bundle_path.join("manifest.json").is_file());
    assert!(bundle_path.join("vectors.svec").is_file());
    assert!(bundle_path.join("index.sidx").is_file());

    let query = Command::new(slate_bin())
        .arg("query")
        .arg(&bundle_path)
        .arg(&query_path)
        .arg("--k")
        .arg("3")
        .output()
        .expect("spawn slate query");
    assert!(query.status.success(), "`slate query` failed");

    let stdout = String::from_utf8(query.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 3, "expected k=3 result lines, got: {stdout:?}");

    // First column of the first line is the nearest neighbour's id == 2.
    let nearest_id: u64 = lines[0]
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(nearest_id, 2, "nearest neighbour should be the exact match");
}

#[test]
fn unknown_backend_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let vectors_path = dir.path().join("vectors.txt");
    let bundle_path = dir.path().join("bundle");
    fs::write(&vectors_path, "1 2 3 4\n5 6 7 8\n").unwrap();

    let build = Command::new(slate_bin())
        .arg("build")
        .arg(&vectors_path)
        .arg(&bundle_path)
        .arg("--backend")
        .arg("nope")
        .status()
        .expect("spawn slate build");
    assert!(!build.success(), "unknown backend should fail the build");
}

/// Build a small bundle and return its directory inside `dir`.
fn build_small_bundle(dir: &std::path::Path) -> std::path::PathBuf {
    let vectors_path = dir.join("vectors.txt");
    let bundle_path = dir.join("bundle");
    fs::write(
        &vectors_path,
        "0 0 0 0\n\
         9 9 9 9\n\
         1 2 3 4\n\
         5 5 5 5\n\
         -1 -2 -3 -4\n",
    )
    .unwrap();
    let build = Command::new(slate_bin())
        .arg("build")
        .arg(&vectors_path)
        .arg(&bundle_path)
        .status()
        .expect("spawn slate build");
    assert!(build.success(), "`slate build` failed");
    bundle_path
}

#[test]
fn bench_reports_cost_and_recall() {
    let dir = tempfile::tempdir().unwrap();
    let bundle_path = build_small_bundle(dir.path());
    let queries_path = dir.path().join("queries.txt");
    fs::write(&queries_path, "1 2 3 4\n5 5 5 5\n").unwrap();

    let bench = Command::new(slate_bin())
        .arg("bench")
        .arg(&bundle_path)
        .arg(&queries_path)
        .arg("--k")
        .arg("3")
        .arg("--profile")
        .arg("hdd")
        .arg("--recall")
        .output()
        .expect("spawn slate bench");
    assert!(bench.status.success(), "`slate bench` failed");

    let stdout = String::from_utf8(bench.stdout).unwrap();
    assert!(stdout.contains("bench: 2 queries"), "report header: {stdout:?}");
    assert!(stdout.contains("storage_fraction"), "missing cost model: {stdout:?}");
    assert!(stdout.contains("recall@3"), "missing recall line: {stdout:?}");
}

#[test]
fn bench_without_recall_omits_recall_line() {
    let dir = tempfile::tempdir().unwrap();
    let bundle_path = build_small_bundle(dir.path());
    let queries_path = dir.path().join("queries.txt");
    fs::write(&queries_path, "1 2 3 4\n").unwrap();

    let bench = Command::new(slate_bin())
        .arg("bench")
        .arg(&bundle_path)
        .arg(&queries_path)
        .arg("--profile")
        .arg("ssd")
        .output()
        .expect("spawn slate bench");
    assert!(bench.status.success(), "`slate bench` failed");

    let stdout = String::from_utf8(bench.stdout).unwrap();
    assert!(stdout.contains("profile=ssd"), "profile not echoed: {stdout:?}");
    assert!(!stdout.contains("recall@"), "recall should be omitted: {stdout:?}");
}
