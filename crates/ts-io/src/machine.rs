//! Machine probe for the memory planner: total / available RAM, cgroup v2 limit, cores.
//! macOS: `hw.memsize`, `host_statistics64` (available = free + inactive + purgeable pages),
//! `hw.perflevel{0,1}.physicalcpu` (P / E cores). Linux: `/proc/meminfo` MemAvailable,
//! cgroup v2 `memory.max - memory.current` along the process's cgroup path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Machine {
    /// physical RAM (bytes)
    pub total: u64,
    /// RAM available without swapping (bytes) (already capped by the cgroup headroom)
    pub avail: u64,
    /// cgroup v2 headroom `max - current` (bytes), if limited
    pub cgroup: Option<u64>,
    /// usable hardware threads (affinity / cgroup cpu quota aware)
    pub cores: usize,
    /// performance / efficiency cores (Apple Silicon)
    pub p_cores: Option<usize>,
    pub e_cores: Option<usize>,
}

pub fn probe() -> Machine {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut m = os::probe(cores);
    if let Some(c) = m.cgroup {
        m.avail = m.avail.min(c);
    }
    m
}

/// `/proc/meminfo` → (MemTotal, MemAvailable) in bytes.
pub fn parse_meminfo(t: &str) -> Option<(u64, u64)> {
    let kb = |key: &str| -> Option<u64> {
        t.lines()
            .find_map(|l| l.strip_prefix(key))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
            .map(|v| v * 1024)
    };
    Some((kb("MemTotal:")?, kb("MemAvailable:")?))
}

/// cgroup v2 path of `/proc/self/cgroup` (`0::/path`).
pub fn parse_cgroup_path(t: &str) -> Option<String> {
    t.lines()
        .find_map(|l| l.strip_prefix("0::"))
        .map(|p| p.trim().to_string())
}

/// `/sys/fs/cgroup` directories from the process's cgroup up to the root.
pub fn cgroup_dirs(path: &str) -> Vec<String> {
    let mut p = path.trim_end_matches('/').to_string();
    let mut v = vec![];
    loop {
        v.push(format!("/sys/fs/cgroup{p}"));
        match p.rfind('/') {
            Some(i) => p.truncate(i),
            None => break,
        }
    }
    v
}

/// Headroom of one cgroup level: `memory.max` (`max` = unlimited) minus `memory.current`.
pub fn cgroup_headroom(max: &str, current: &str) -> Option<u64> {
    let m = max.trim();
    if m == "max" {
        return None;
    }
    let (m, c) = (m.parse::<u64>().ok()?, current.trim().parse::<u64>().ok()?);
    Some(m.saturating_sub(c))
}

#[cfg(target_os = "linux")]
mod os {
    use super::*;
    pub fn probe(cores: usize) -> Machine {
        let rd = |p: &str| std::fs::read_to_string(p).ok();
        let (total, avail) = rd("/proc/meminfo")
            .and_then(|t| parse_meminfo(&t))
            .unwrap_or((0, 0));
        // tightest headroom along the cgroup path (limits are hierarchical)
        let cg = rd("/proc/self/cgroup")
            .and_then(|t| parse_cgroup_path(&t))
            .map(|p| cgroup_dirs(&p))
            .unwrap_or_default()
            .iter()
            .filter_map(|d| {
                cgroup_headroom(
                    &rd(&format!("{d}/memory.max"))?,
                    &rd(&format!("{d}/memory.current"))?,
                )
            })
            .min();
        Machine {
            total,
            avail,
            cgroup: cg,
            cores,
            p_cores: None,
            e_cores: None,
        }
    }
}

#[cfg(target_os = "macos")]
mod os {
    use super::*;
    use std::ffi::CString;

    fn sysctl_u64(name: &str) -> Option<u64> {
        let c = CString::new(name).ok()?;
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: valid name, out buffer of `len` bytes
        let r = unsafe {
            libc::sysctlbyname(
                c.as_ptr(),
                &mut v as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (r == 0).then_some(if len == 4 { v & 0xffff_ffff } else { v })
    }

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(host: u32, flavor: i32, info: *mut u32, count: *mut u32) -> i32;
    }

    pub fn probe(cores: usize) -> Machine {
        let total = sysctl_u64("hw.memsize").unwrap_or(0);
        let page = sysctl_u64("hw.pagesize").unwrap_or(16384);
        // vm_statistics64 (HOST_VM_INFO64 = 4, 38 natural_t): [0] free (incl. speculative),
        // [2] inactive, [22] purgeable
        let mut b = [0u32; 64];
        let mut cnt: u32 = 38;
        // SAFETY: buffer of 64 natural_t >= HOST_VM_INFO64_COUNT
        let r = unsafe { host_statistics64(mach_host_self(), 4, b.as_mut_ptr(), &mut cnt) };
        let avail = if r == 0 {
            (b[0] as u64 + b[2] as u64 + b[22] as u64) * page
        } else {
            total / 2
        };
        let us = |n: &str| sysctl_u64(n).map(|v| v as usize);
        Machine {
            total,
            avail: avail.min(total),
            cgroup: None,
            cores,
            p_cores: us("hw.perflevel0.physicalcpu"),
            e_cores: us("hw.perflevel1.physicalcpu"),
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod os {
    use super::*;
    pub fn probe(cores: usize) -> Machine {
        Machine {
            total: 0,
            avail: 0,
            cgroup: None,
            cores,
            p_cores: None,
            e_cores: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_parsers() {
        let mi = "MemTotal:       16303088 kB\nMemFree:  1000 kB\nMemAvailable:   12001244 kB\n";
        assert_eq!(parse_meminfo(mi), Some((16303088 * 1024, 12001244 * 1024)));
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
        assert_eq!(
            parse_cgroup_path("0::/user.slice/job-1.scope\n").as_deref(),
            Some("/user.slice/job-1.scope")
        );
        assert_eq!(parse_cgroup_path("1:name=systemd:/x\n"), None);
        assert_eq!(
            cgroup_dirs("/a/b"),
            ["/sys/fs/cgroup/a/b", "/sys/fs/cgroup/a", "/sys/fs/cgroup"]
        );
        assert_eq!(cgroup_dirs("/"), ["/sys/fs/cgroup"]);
        assert_eq!(cgroup_headroom("max\n", "123\n"), None);
        assert_eq!(
            cgroup_headroom("6442450944\n", "1073741824\n"),
            Some(5368709120)
        );
        assert_eq!(cgroup_headroom("100", "200"), Some(0));
    }

    #[test]
    fn probe_is_sane() {
        let m = probe();
        assert!(m.cores >= 1);
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            assert!(m.total > 0 && m.avail > 0 && m.avail <= m.total, "{m:?}");
        }
    }
}
