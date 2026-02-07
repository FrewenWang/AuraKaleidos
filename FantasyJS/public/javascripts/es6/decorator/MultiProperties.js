export function MultiProperties(...list) {
    return function (target) {
        Object.assign(target.prototype, ...list)
    }
}
