//! Test file to understand Extism 1.13 CurrentPlugin API

use extism::{Function, UserData, PTR, Val, CurrentPlugin};

pub fn test_host_function() -> Function {
    Function::new(
        "test_fn",
        [PTR],
        [PTR],
        UserData::new(()),
        |plugin: &mut CurrentPlugin, inputs: &[Val], outputs: &mut [Val], _user_data: UserData<()>| {
            // What methods does plugin have?
            // Try: plugin.memory_get_val
            // Try: plugin.input
            // Try: plugin.memory_new
            Ok(())
        },
    )
}
