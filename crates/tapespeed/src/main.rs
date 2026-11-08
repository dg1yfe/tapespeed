use tapespeed::{cli, pipeline, sub, survey};

#[cfg(feature = "memprof")]
#[global_allocator]
static ALLOC: ts_io::prof::Counting = ts_io::prof::Counting;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if let Some(code) = sub::dispatch(&argv) {
        std::process::exit(code);
    }
    let a = cli::parse_from(&argv);
    if let Some(n) = a.threads.filter(|&n| n > 0) {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }
    let r = if a.survey {
        survey::main_survey(&a)
    } else {
        if let Err(msg) = cli::check_required(&a) {
            eprintln!("{}\ntapespeed: error: {msg}", cli::usage());
            std::process::exit(2);
        }
        pipeline::main_run(&a)
    };
    if let Err(e) = r {
        eprintln!("{}", e.message());
        std::process::exit(e.code());
    }
}
