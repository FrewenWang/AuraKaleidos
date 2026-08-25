//
//  AliceSwiftIOSAppTests.swift
//  AliceSwiftIOSAppTests
//
//  Created by Frewen.Wong on 2023/2/15.
//

import XCTest
@testable import AliceSwiftIOSApp

final class AliceSwiftIOSAppTests: XCTestCase {
    func testViewControllerLoadsItsView() {
        let viewController = ViewController()

        viewController.loadViewIfNeeded()

        XCTAssertTrue(viewController.isViewLoaded)
        XCTAssertNotNil(viewController.view)
    }
}
