//! Minimal JSON value/emitter: ASCII-only output (`\uXXXX` escapes), NaN/Infinity tokens,
//! shortest round-trip floats, object keys in insertion order (part of the file format),
//! compact (`, ` / `: `) or indented output.
use crate::numfmt::repr_shortest;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn obj() -> Value {
        Value::Obj(Vec::new())
    }

    /// Append a key (builder style); panics if `self` is not an object.
    pub fn with(mut self, k: &str, v: impl Into<Value>) -> Value {
        self.push(k, v);
        self
    }

    pub fn push(&mut self, k: &str, v: impl Into<Value>) {
        match self {
            Value::Obj(o) => o.push((k.to_string(), v.into())),
            _ => panic!("Value::push on non-object"),
        }
    }

    pub fn get(&self, k: &str) -> Option<&Value> {
        match self {
            Value::Obj(o) => o.iter().find(|(kk, _)| kk == k).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Compact text (`indent = None`: `, ` and `: ` separators, one line) or one item per line
    /// indented by `n` spaces per level.
    pub fn dumps(&self, indent: Option<usize>) -> String {
        let mut s = String::new();
        write_value(&mut s, self, indent, 0);
        s
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::Int(v as i64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::Arr(v.into_iter().map(Into::into).collect())
    }
}

/// Float to JSON text: [`repr_shortest`], but `NaN` / `Infinity` / `-Infinity`.
pub fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x == f64::INFINITY {
        "Infinity".into()
    } else if x == f64::NEG_INFINITY {
        "-Infinity".into()
    } else {
        repr_shortest(x)
    }
}

/// JSON string literal, ASCII only: everything outside printable ASCII is escaped (`\n`, `\t`,
/// ..., else `\uXXXX`, surrogate pairs above U+FFFF).
pub fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            ' '..='~' => o.push(c),
            _ => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    o.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    o.push('"');
    o
}

fn nl(s: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        s.push('\n');
        s.push_str(&" ".repeat(n * level));
    }
}

fn write_value(s: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => s.push_str("null"),
        Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Value::Int(i) => s.push_str(&i.to_string()),
        Value::Float(x) => s.push_str(&json_float(*x)),
        Value::Str(t) => s.push_str(&json_str(t)),
        Value::Arr(a) if a.is_empty() => s.push_str("[]"),
        Value::Obj(o) if o.is_empty() => s.push_str("{}"),
        Value::Arr(a) => {
            s.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                    if indent.is_none() {
                        s.push(' ');
                    }
                }
                nl(s, indent, level + 1);
                write_value(s, e, indent, level + 1);
            }
            nl(s, indent, level);
            s.push(']');
        }
        Value::Obj(o) => {
            s.push('{');
            for (i, (k, e)) in o.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                    if indent.is_none() {
                        s.push(' ');
                    }
                }
                nl(s, indent, level + 1);
                s.push_str(&json_str(k));
                s.push_str(": ");
                write_value(s, e, indent, level + 1);
            }
            nl(s, indent, level);
            s.push('}');
        }
    }
}
