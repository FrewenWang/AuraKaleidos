function decoratorTest(target) {
    target.isTestable = true;
}

@decoratorTest()
class MyDecoratorClass {

}

console.log(MyDecoratorClass.isTestable);
