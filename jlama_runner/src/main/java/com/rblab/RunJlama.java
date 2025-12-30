package com.rblab;

import com.github.tjake.jlama.cli.JlamaCli;

public class RunJlama {
    public static void main(String[] args) {
        // Delegate to Jlama's main CLI entry point
        // We expect args to be passed like "restapi", "path/to/model"
        JlamaCli.main(args);
    }
}
