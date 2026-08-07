#include <gtest/gtest.h>

#include "aura/utils/object_pool_v2.hpp"

namespace {

struct Value {
    explicit Value(int initial) : value(initial) { }
    int value;
};

TEST(ObjectPoolTest, ReusesReleasedObject) {
    aura::aura_utils::ObjectPool pool;
    pool.Create<Value, int>(2);

    auto first = pool.Get<Value, int>(1);
    Value *address = first.get();
    first.reset();

    auto second = pool.Get<Value, int>(2);
    EXPECT_EQ(address, second.get());
    EXPECT_EQ(2, second->value);
}

TEST(ObjectPoolTest, PointerCanOutlivePool) {
    std::shared_ptr<Value> survivor;
    {
        aura::aura_utils::ObjectPool pool;
        pool.Create<Value, int>(1);
        survivor = pool.Get<Value, int>(7);
    }

    ASSERT_NE(nullptr, survivor);
    EXPECT_EQ(7, survivor->value);
    survivor.reset();
}

} // namespace
