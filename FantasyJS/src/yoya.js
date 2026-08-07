import * as RemoteKeyCode from '../keyCoderemoteKeyCode'
import * as KeyboardKeyCode from './keyboardKeyCode'

const htmlFontSize = parseFloat(window.document.querySelector('html').style.fontSize) // 1rem = 1px / htmlFontSize

function YoyaCreateElement (eleType, props = {}) {
  if (typeof (eleType) === 'function') {
    const children = []
    for (let i = 2; i < arguments.length; i++) {
      children.push(arguments[i])
    }
    return eleType({ ...props, children: children })
  }
  const htmlEle = document.createElement(eleType)

  // if ('tabIndex' in props) {
  //     htmlEle.tabIndex = props.tabIndex;
  // }
  if ('className' in props) {
    htmlEle.className = props.className
  }
  if ('tabIndex' in props) {
    htmlEle.tabIndex = props.tabIndex
  }
  if ('id' in props) {
    htmlEle.id = props.id
  }
  if ('ref' in props) {
    props.ref.node = htmlEle
  }
  if ('style' in props) {
    for (const i in props.style) {
      htmlEle.style[i] = props.style[i]
    }
  }
  // if ('dir' in props) { // 仅用于阿拉伯等从右向左读的文字
  //   htmlEle.dir = props.dir
  // }
  // 给需要滚动的元素父元素设置自定义scrollable属性，元素要设置width。
  if ('scrollable' in props) {
    htmlEle.setAttribute('scrollable', props.scrollable)
  }
  // 给需要滚动的元素设置自定义scrollElement属性，元素不可设置width,display，滚动元素自身距离
  if ('scrollElement' in props) {
    htmlEle.setAttribute('scrollElement', props.scrollElement)
  }
  if ('deepLink' in props) {
    htmlEle.setAttribute('deepLink', props.deepLink)
  }

  for (let i = 2; i < arguments.length; i++) {
    if (arguments[i] instanceof Array) {
      for (let j = 0; j < arguments[i].length; j++) {
        if ((typeof arguments[i]) === 'string') {
          htmlEle.innerHTML = arguments[i][j]
          continue
        }
        if ((arguments[i] === null) || (arguments[i]) === undefined) {
          continue
        }
        htmlEle.appendChild(arguments[i][j])
      }
      continue
    }
    if ((typeof arguments[i]) === 'string') {
      htmlEle.innerHTML = arguments[i]
      continue
    }
    if ((arguments[i] === null) || (arguments[i]) === undefined || (arguments[i] === false)) {
      continue
    }
    htmlEle.appendChild(arguments[i])
  }

  // if ('onRender' in props) {
  //     props.onRender();
  // }

  return htmlEle
}

function YoyaCreateItemList (params = {}) {
  let props = {
    dirH: true,
    cycle: true,
    transformEnable: true,
    transition: 0.35,
    isTransitionEnable: true,
    padding: 0,
    foresee: 0.7,
    className: 'list-container',
    prevKeys: [],
    nextKeys: []
  }
  props = { ...props, ...params }
  if (props.dirH) {
    props.prevKeys.push(RemoteKeyCode.KK_KEY_ArrowLeft)
    props.nextKeys.push(RemoteKeyCode.KK_KEY_ArrowRight)

    props.prevKeys.push(KeyboardKeyCode.KB_KEY_ArrowLeft)
    props.nextKeys.push(KeyboardKeyCode.KB_KEY_ArrowRight)

    props.prevKeys.push(RemoteKeyCode.KK_KEY_KEYPAD_VOLUME_MINUS)
    props.nextKeys.push(RemoteKeyCode.KK_KEY_KEYPAD_VOLUME_PLUS)
  } else {
    props.prevKeys.push(RemoteKeyCode.KK_KEY_ArrowUp)
    props.nextKeys.push(RemoteKeyCode.KK_KEY_ArrowDown)

    props.prevKeys.push(KeyboardKeyCode.KB_KEY_ArrowUp)
    props.nextKeys.push(KeyboardKeyCode.KB_KEY_ArrowDown)

    props.prevKeys.push(RemoteKeyCode.KK_KEY_KEYPAD_CHANNEL_PLUS)
    props.nextKeys.push(RemoteKeyCode.KK_KEY_KEYPAD_CHANNEL_MINUS)
  }

  function getFirstFocusableIndex () {
    let index = 0
    while (component.rootNode.childNodes[index].tabIndex !== -1) {
      index++
    }
    return index
  }
  function getLastFocusableIndex () {
    let index = component.rootNode.childNodes.length - 1
    while (component.rootNode.childNodes[index].tabIndex !== -1) {
      index--
    }
    return index
  }
  function getNextFocusableIndex (currentIndex) {
    let netxValue = currentIndex
    do {
      if (netxValue >= getLastFocusableIndex()) {
        netxValue = props.cycle ? getFirstFocusableIndex() : getLastFocusableIndex()
      } else {
        netxValue = netxValue + 1
      }
    } while (component.rootNode.childNodes[netxValue].tabIndex !== -1)
    return netxValue
  }
  function getPrevFocusableIndex (currentIndex) {
    let netxValue = currentIndex
    do {
      if (netxValue <= getFirstFocusableIndex()) {
        netxValue = props.cycle ? getLastFocusableIndex() : getFirstFocusableIndex()
      } else {
        netxValue = netxValue - 1
      }
    } while (component.rootNode.childNodes[netxValue].tabIndex !== -1)
    return netxValue
  }
  function focusPrevIndex () {
    component.focusMoveDir = 'prev'
    component.setFocusIndex(getPrevFocusableIndex(component.focusIndex), true)
  }
  function focusNextIndex () {
    component.focusMoveDir = 'next'
    component.setFocusIndex(getNextFocusableIndex(component.focusIndex), true)
  }

  function handleOnKeyDown (event) {
    if (props.prevKeys.indexOf(event.keyCode) !== -1) {
      focusPrevIndex()
      event.stopPropagation()
    } else if (props.nextKeys.indexOf(event.keyCode) !== -1) {
      focusNextIndex()
      event.stopPropagation()
    }
  }

  function calculateTransform (isTransitionEnable = true) {
    if (!props.transformEnable) {
      return
    }
    const node = component.rootNode
    const focusIndex = component.focusIndex
    const foresee = props.foresee ? props.foresee : 0.5
    if (node === null) {
      console.log('First render, ref node is null!s')
      return
    }

    let transform = 0
    let currentTransform = 0
    if (window.getComputedStyle(node, null).transform !== 'none') {
      currentTransform = node.style.transform.split('(')[1].split('rem')[0] - 0
    }

    const parentWidth = node.parentNode.offsetWidth / htmlFontSize
    const parentHeight = node.parentNode.offsetHeight / htmlFontSize
    const currentItemLeft = node.childNodes[focusIndex].offsetLeft / htmlFontSize
    const currentItemWidth = node.childNodes[focusIndex].offsetWidth / htmlFontSize
    const currentItemTop = node.childNodes[focusIndex].offsetTop / htmlFontSize
    const currentItemHeight = node.childNodes[focusIndex].offsetHeight / htmlFontSize
    if (focusIndex <= getFirstFocusableIndex()) {
      transform = props.padding
    } else if (focusIndex >= getLastFocusableIndex()) {
      if (props.dirH
        ? (currentItemLeft + currentItemWidth + currentTransform + props.padding > parentWidth)
        : (currentItemTop + currentItemHeight + currentTransform - props.padding > parentHeight)
      ) {
        transform = - (
          props.dirH
          ? currentItemLeft + currentItemWidth + props.padding - parentWidth
          : currentItemTop + currentItemHeight + props.padding - parentHeight
        )
      } else {
        transform = currentTransform
      }
    } else {
      if (props.dirH ? (currentItemLeft + currentTransform < props.padding) : (currentItemTop + currentTransform < props.padding)) {
        transform = - (
          props.dirH
          ? currentItemLeft + props.padding - node.childNodes[getPrevFocusableIndex(focusIndex)].offsetWidth / htmlFontSize * foresee
          : currentItemTop + props.padding - node.childNodes[getPrevFocusableIndex(focusIndex)].offsetHeight / htmlFontSize * foresee
        )
      } else if (props.dirH ? (currentItemLeft + currentItemWidth + currentTransform + props.padding > parentWidth) : (currentItemTop + currentItemHeight + currentTransform - props.padding > parentHeight)) {
        transform = - (
          props.dirH
          ? currentItemLeft + currentItemWidth + node.childNodes[getNextFocusableIndex(focusIndex)].offsetWidth / htmlFontSize * foresee + props.padding - parentWidth
          : currentItemTop + currentItemHeight + node.childNodes[getNextFocusableIndex(focusIndex)].offsetHeight / htmlFontSize * foresee + props.padding - parentHeight
        )
      } else {
        transform = currentTransform
      }
    }

    node.style.transitionTimingFunction = 'linear'
    node.style.transform = (props.dirH ? 'translateX(' : 'translateY(') + transform + 'rem)'
    node.style.transitionDuration = (props.transition ? props.transition : 0.5) + 's'
    if (isTransitionEnable === false) {
      node.style.transitionDuration = '0s'
    }
    component.transform = transform
  }

  const component = {
    // data
    focusIndex: 0,
    childs: [],
    father: null,
    rootNode: null,
    onFocusIndexChange: [],
    transform: props.transform ? props.transform : props.padding,
    focusMoveDir: 'null',
    isTransitionEnable: props.isTransitionEnable,

    // actions
    focus: function () {
      if (this.childs[this.focusIndex].item.focusStatus === 'nofocus') {
        this.focusIndex = getNextFocusableIndex(this.focusIndex)
      }

      this.childs[this.focusIndex].item.focus({ preventScroll: true })
      calculateTransform(this.isTransitionEnable)
    },

    setFocusStatus: function (state) {},
    /*
        *   immediately value in:
        *       true（default）: 对应元素立马获焦
        *       false: 只修改focusIndex的值，只是逻辑意义上的修改记录焦点的数据
        */
    setFocusIndex: function (index = 0, immediately = true) {
      if (index >= this.childs.length) {
        console.log('[yoya] Error: Out of bounds!')
        return
      }
      if (!immediately) {
        if ((this.childs[this.focusIndex].type === 'component') &&
                (this.childs[this.focusIndex].item.focusStatus === 'semifocus')) {
          this.childs[this.focusIndex].item.setFocusStatus('unfocus')
        }
        if (this.childs[index].type === 'component' && this.childs[index].item.focusStatus !== 'nofocus') {
          this.childs[index].item.setFocusStatus('semifocus')
        }
      }
      if (this.focusIndex === index) {
        // console.log("Info: The focusIndex already is: " + index);
        return
      }
      const lastIndex = this.focusIndex
      this.focusIndex = index
      if (immediately) {
        if (this.childs[index].type === 'html') {
          this.childs[index].item.focus({ preventScroll: true })
          calculateTransform(this.isTransitionEnable)
        } else {
          this.childs[index].item.focus()
          calculateTransform(this.isTransitionEnable)
        }
      }
      this.onFocusIndexChange.forEach((item) => {
        item(this.focusIndex, this, lastIndex)
      })
    },

    addFocusChangeEvent: function () {
      this.onFocusIndexChange.forEach((item) => {
        item(this.focusIndex, this, this.focusIndex)
      })
    },

    setTransform: function (transform) {
      this.transform = transform
      this.rootNode.style.transform = (props.dirH ? 'translateX(' : 'translateY(') + transform + 'rem)'
    },
    // type value in: "html" "component"
    addChild: function (child, type, key = 'undefined') {
      if ((type !== 'html') && (type !== 'component')) {
        console.log('Error: Unkown type: ' + type)
        return
      }
      child.father = this
      this.childs.push({
        key: key + '',
        type: type,
        item: child
      })
      if (type === 'html') {
        this.rootNode.appendChild(child)
      } else {
        this.rootNode.appendChild(child.rootNode)
      }
    },
    getChildIndex: function (childItem) {
      for (let i = 0; i < this.childs.length; i++) {
        if (childItem === this.childs[i].item) {
          return i
        }
      }
    },
    getChildByIndex: function (index) {
      if (index >= this.childs.length) {
        console.log('Error: Child length: ' + this.childs.length + ', you get index: ' + index)
        return null
      }
      return this.childs[index].item
    },
    removeChildByIndex: function (index) {
      if (index >= this.childs.length) {
        console.log('Error: Child length: ' + this.childs.length + ', you remove index: ' + index)
        return
      }

      if (this.childs[index].type === 'html') {
        this.rootNode.removeChild(this.childs[index].item)
      } else {
        this.rootNode.removeChild(this.childs[index].item.rootNode)
      }

      this.childs.splice(index, 1)
    },
    getChildByKey: function (key) {
      for (let i = 0, len = this.childs.length; i < len; i++) {
        if (this.childs[i].key === key) {
          return this.childs[i].item
        }
      }
      console.log('Error: Not font child key: ' + key)
      return null
    },
    removeChildByKey: function (key) {
      for (let i = 0, len = this.childs.length; i < len; i++) {
        if (this.childs[i].key === key) {
          if (this.childs[i].type === 'html') {
            this.rootNode.removeChild(this.childs[i].item)
          } else {
            this.rootNode.removeChild(this.childs[i].item.rootNode)
          }

          this.childs.splice(i, 1)
          return
        }
      }
      console.log('Error: Not font child key: ' + key)
    },
    clearChild: function () {
      this.rootNode.innerHTML = ''
      this.childs = []
    },
    // Registration event
    addEventMethod: function (eventName, eventMethod) {
      this[eventName].push(eventMethod)
    },
    removeEventMethod: function (eventName, eventMethod) {
      this[eventName].forEach((item, index) => {
        if (item === eventMethod) {
          this[eventName].splice(index, 1)
        }
      })
    },

    // getData
    getFocusIndex: function () {
      return this.focusIndex
    },
    // getFocusStatus: function () {
    //     return this.focusStatus;
    // },

    // render
    render: function () {
      return this.rootNode
    }
  }

  component.rootNode = document.createElement('div')
  component.rootNode.innerHTML = ''
  component.rootNode.className = props.className
  component.rootNode.tabIndex = -1
  component.rootNode.style.transform = (props.dirH ? 'translateX(' : 'translateY(') + component.transform + 'rem)'
  component.rootNode.addEventListener('keydown', handleOnKeyDown)

  return component
}

function YoyaCreateItem (child, childProps = {}) {
  const component = {
    // data
    focusStatus: 'unfocus',
    rootNode: null,
    father: null,
    onFocusStatusChange: [],

    // actions
    focus: function () {
      if (this.focusStatus === 'nofocus') {
        console.log('Error: Component focusStatus is "nofocus"! get focus fail!')
        return
      }
      this.rootNode.focus({ preventScroll: true })
    },
    // state value in: "unfocus" "focus" "semifocus" "nofocus"
    setFocusStatus: function (state) {
      if ((state !== 'unfocus') && (state !== 'focus') && (state !== 'semifocus') && (state !== 'nofocus')) {
        console.log('Error: Unkown state: ' + state)
        return
      }
      if (this.focusStatus === state) {
        // console.log("Info: The focusStatus already is: " + state);
        return
      }
      if (this.rootNode) {
        if (state === 'nofocus') {
          this.rootNode.tabIndex = 0
        } else {
          this.rootNode.tabIndex = -1
        }
      }
      const lastStatus = this.focusStatus
      this.focusStatus = state
      this.onFocusStatusChange.forEach((item) => {
        item(this.focusStatus, this, lastStatus)
      })
    },
    // Registration event
    addEventMethod: function (eventName, eventMethod) {
      this[eventName].push(eventMethod)
    },
    removeEventMethod: function (eventName, eventMethod) {
      this[eventName].forEach((item, index) => {
        if (item === eventMethod) {
          this[eventName].splice(index, 1)
        }
      })
    },

    // getData
    getFocusStatus: function () {
      return this.focusStatus
    },

    // render
    render: function () {
      return this.rootNode
    }
  }

  if (typeof child === 'function') {
    component.rootNode = child(childProps)
  } else {
    component.rootNode = child
  }
  component.rootNode.tabIndex = (component.focusStatus === 'nofocus') ? 0 : -1
  function handleBlur () {
    if (component.focusStatus === 'focus') {
      if ((component.father !== null) && (component.father.childs[component.father.focusIndex].item === component)) {
        component.setFocusStatus('semifocus')
      } else {
        component.setFocusStatus('unfocus')
      }
    }
  }
  function handleFocus () {
    component.setFocusStatus('focus')
  }
  component.rootNode.addEventListener('blur', handleBlur)
  component.rootNode.addEventListener('focus', handleFocus)

  return component
}

export function YoyaRender (component, node) {
  node.appendChild(component.render())
}

export const Yoya = {
  /**
     * features: 创建一个html元素，类似于document.createElement接口，可递归或嵌套使用以创建
     *       出复杂的html元素
     * params: eleType: html元素名或返回值为html元素的函数
     *         props: 一个json对象，包含元素属性，目前支持（className, id, ref, style）
     *         ...: 后可接任意多个此接口（createElement）的调用，作为当前元素的子元素，
     * return: html元素对象
     */
  createElement: YoyaCreateElement,

  /**
     * features: 创建一个列表容器，自带焦点移动，transform计算功能
     * params: 传入一个json对象，可设置的参数如下（括号内为缺省值）：
     *      dirH: (true) 列表的方向，true表示横向，false表示纵向
     *      cycle: (true) 焦点是否循环移动
     *      transition: (0.5) transform过渡时间
     *      padding: (0) 列表头部和尾部的空白填充
     *      foresee: (0.5) 焦点移动过程中，计算transform时，下一个元素显示出的比例
     *      className: ('list-container') 此接口会创建一个div元素用于包含其子元素，此为创建的元素的class属性值
     *      prevKeys: [] 自定义响应往前移动焦点的按键，yoya会自动加上方向键上或左
     *      nextKeys: [] 自定义响应往后移动焦点的按键，yoya会自动加上方向键下或右
     * return: 返回一个对象，对象中包含一系列操作接口：
     *      {
     *      focus: function ()
     *          获焦，因为为list，逻辑上不可以获焦，所以实际是它的子元素获焦
     *
     *      setFocusIndex: function (index = 0, immediately = true)
     *          设置第index个子元素获焦，immediately为true表示立马生效，false表示只是内部记录获焦index值的修改，并不产生实际上的焦点变化
     *
     *      addChild: function (child, type, key = 'undefined')
     *          给列表添加一个子组件/元素，child取值为html元素对象或者由createItemList、createItem接口返回的组件对象，对应type取值为
     *     （"html" "component"），key的值应该为一个string，且同一个类型的所有child的key值不应该重复
     *
     *      getChildByIndex: function (index)
     *          通过索引获取某个子元素的对象
     *
     *      removeChildByIndex: function (index)
     *          通过索引删除子元素
     *
     *      getChildByKey: function (key)
     *          通过key获取某个子元素对象
     *
     *      removeChildByKey: function (key)
     *          通过key删除某个子元素对象
     *
     *      clearChild: function ()
     *          清空子元素
     *
     *      addEventMethod: function (eventName, eventMethod)
     *      removeEventMethod: function (eventName, eventMethod)
     *          添加/删除事件监听函数，eventName：事件类型，支持的事件类型（onFocusIndexChange）；eventMethod事件响应函数，会传入当前对象作为参数
     *
     *      getFocusIndex: function ()
     *          获取当前获焦子元素index
     *      }
     */
  createItemList: YoyaCreateItemList,

  /**
     * features: 创建一个可获焦组件，像常见的button、imageButton、menu菜单元素等组件都可以用此接口创建
     * params:
     *      child: 创建组件结构的函数，返回值应该为html元素，或者html元素，当为html元素时childProps无效
     *      childProps: 创建组件结构函数的传入参数，在组件渲染时，会调用childCreater函数并传入childProps值
     * return: 返回一个对象，对象中包含一系列对所创建的组件的操作接口
     *      {
     *      focus: function ()
     *          获焦
     *
     *      setFocusStatus: function (state)
     *          设置焦点状态，stat取值范围：（"unfocus" "focus" "semifocus" "nofocus"），对应意思为（不获焦、获焦、半获焦、不可获焦）
     *
     *      addEventMethod: function (eventName, eventMethod)
     *      removeEventMethod: function (eventName, eventMethod)
     *          添加/删除事件监听函数，eventName：事件类型，支持的事件类型（onFocusStatusChange）；eventMethod事件响应函数，会传入当前对象作为参数
     *
     *      getFocusStatus: function ()
     *          获取组件焦点状态
     *      }
     */
  createItem: YoyaCreateItem,

  /**
     * features: 渲染createItemList、createItem接口创建的组件
     * params:
     *      component: 被渲染的组件对象
     *      node: 渲染的的目标节点，值应该为html元素对象
     * return: undefined
     */
  render: YoyaRender

  /**
     * 注意：接口的基本使用也可参考demo项目代码
     */
}
