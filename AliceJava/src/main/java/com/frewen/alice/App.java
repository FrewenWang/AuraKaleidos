package com.frewen.alice;

public final class App {
    private App() {}

    public static String greeting(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("name must not be blank");
        }
        return "Hello, " + name + "!";
    }

    public static void main(String[] args) {
        String name = args.length == 0 ? "AuraKaleidos" : args[0];
        System.out.println(greeting(name));
    }
}
